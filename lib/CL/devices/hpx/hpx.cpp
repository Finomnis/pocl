/* OpenCL hpx device implementation.

   Copyright (c) 2011-2012 Universidad Rey Juan Carlos and
                           Pekka Jääskeläinen / Tampere Univ. of Technology
                      2014 Martin Stumpf
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/


extern "C" {
    #include "pocl-hpx.h"
    #include "install-paths.h"
    #include "pocl_runtime_config.h"
    #include "utlist.h"
    #include "cpuinfo.h"
    #include "topology/pocl_topology.h"
    #include "common.h"
    #include "config.h"
    #include "devices.h"
    #include "pocl_util.h"
    #include "pocl_mem_management.h"
}


#include <hpx/config.hpp>
#include <hpx/hpx.hpp>
#include <hpx/parallel/algorithm.hpp>

#include <boost/iterator/counting_iterator.hpp>

#include <vector>
#include <cassert>

#ifdef CUSTOM_BUFFER_ALLOCATOR

extern "C" {
    #include "bufalloc.h"
    #include <../dev_image.h>
}
/* Instead of mallocing a buffer size for a region, try to allocate 
   this many times the buffer size to hopefully avoid mallocs for 
   the next buffer allocations.
   
   Falls back to single multiple allocation if fails to allocate a
   larger region. */
#define ALLOCATION_MULTIPLE 32

/* To avoid memory hogging in case of larger buffers, limit the
   extra allocation margin to this number of megabytes.

   The extra allocation should be done to avoid repetitive calls and
   memory fragmentation for smaller buffers only. 
 */
#define ADDITIONAL_ALLOCATION_MAX_MB 100

/* Whether to immediately free a region in case the last chunk was
   deallocated. If 0, it can reuse the same region over multiple kernels. */
#define FREE_EMPTY_REGIONS 0

/* CUSTOM_BUFFER_ALLOCATOR */
#endif

#define COMMAND_LENGTH 2048
#define WORKGROUP_STRING_LENGTH 128

/* The name of the environment variable used to force a certain max thread count
   for the thread execution. */
#define THREAD_COUNT_ENV "POCL_MAX_PTHREAD_COUNT"

typedef struct thread_arguments thread_arguments;

struct thread_arguments
{
    void *data; //? no idea what this is for
    // pointer to global pocl context
    struct pocl_context *pc_global;
    // array of local pocl contexts
    struct pocl_context **pc_local;
    pocl_workgroup workgroup;
    // one set of arguments per thread. (arguments is void**)
    struct pocl_argument *kernel_args;
    void*** wg_arguments;
    char* initialized;
    cl_kernel kernel;
    unsigned int device;
};

typedef struct _mem_regions_management{
  ba_lock_t mem_regions_lock;
  struct memory_region *mem_regions;
} mem_regions_management;

struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  /* Loaded kernel dynamic library handle. */
  lt_dlhandle current_dlhandle;

#ifdef CUSTOM_BUFFER_ALLOCATOR
  /* Lock for protecting the mem_regions linked list. Held when new mem_regions
     are created or old ones freed. */
  mem_regions_management* mem_regions;
#endif

};


static int get_max_thread_count();
static void workgroup_thread (void* p, size_t gid_x, size_t gid_y, size_t gid_z);

void
pocl_hpx_init_device_ops(struct pocl_device_ops *ops)
{
  pocl_basic_init_device_ops(ops);

  ops->device_name = "HPX";

  /* implementation */
  ops->probe = pocl_hpx_probe;
  ops->init_device_infos = pocl_hpx_init_device_infos;
  ops->uninit = pocl_hpx_uninit;
  ops->init = pocl_hpx_init;
  ops->alloc_mem_obj = pocl_hpx_alloc_mem_obj;
  ops->free = pocl_hpx_free;
  ops->read = pocl_hpx_read;
  ops->write = pocl_hpx_write;
  ops->copy = pocl_hpx_copy;
  ops->copy_rect = pocl_hpx_copy_rect;
  ops->run = pocl_hpx_run;
  ops->compile_submitted_kernels = pocl_basic_compile_submitted_kernels;

}

unsigned int
pocl_hpx_probe(struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);
  /* Env was not specified, default behavior was to use 1 hpx device */
  if(env_count < 0)
    return 1;

  return env_count;
}

void
pocl_hpx_init_device_infos(struct _cl_device_id* dev)
{
  pocl_basic_init_device_infos(dev);

  dev->type = CL_DEVICE_TYPE_CPU;
  /* This could be SIZE_T_MAX, but setting it to INT_MAX should suffice, */
  /* and may avoid errors in user code that uses int instead of size_t */
  dev->max_work_item_sizes[0] = 1024;
  dev->max_work_item_sizes[1] = 1024;
  dev->max_work_item_sizes[2] = 1024;

}

void
pocl_hpx_init (cl_device_id device, const char* parameters)
{
  struct data *d; 
  static mem_regions_management* mrm = NULL;

  // TODO: this checks if the device was already initialized previously.
  // Should we instead have a separate bool field in device, or do the
  // initialization at library startup time with __attribute__((constructor))?
  if (device->data!=NULL)
    return;  

  d = (struct data *) malloc (sizeof (struct data));
  
  d->current_kernel = NULL;
  d->current_dlhandle = 0;

  device->data = d;
#ifdef CUSTOM_BUFFER_ALLOCATOR  
  if (mrm == NULL)
    {
      mrm = (mem_regions_management*) malloc (sizeof (mem_regions_management));
      BA_INIT_LOCK (mrm->mem_regions_lock);
      mrm->mem_regions = NULL;
    }
  d->mem_regions = mrm;
#endif  

  device->address_bits = sizeof(void*) * 8;

  /* Use the minimum values until we get a more sensible 
     upper limit from somewhere. */
  device->max_read_image_args = device->max_write_image_args = 128;
  device->image2d_max_width = device->image2d_max_height = 8192;
  device->image3d_max_width = device->image3d_max_height = device->image3d_max_depth = 2048;
  device->max_samplers = 16;  
  device->max_constant_args = 8;

  device->min_data_type_align_size = device->mem_base_addr_align = MAX_EXTENDED_ALIGNMENT;

  /* Note: The specification describes identifiers being delimited by
     only a single space character. Some programs that check the device's
     extension  string assume this rule. Future extension additions should
     ensure that there is no more than a single space between
     identifiers. */

#if SIZEOF_DOUBLE == 8
#define DOUBLE_EXT "cl_khr_fp64 "
#else
#define DOUBLE_EXT 
#endif

#if SIZEOF___FP16 == 2
#define HALF_EXT "cl_khr_fp16 "
#else
#define HALF_EXT
#endif

  device->extensions = DOUBLE_EXT HALF_EXT "cl_khr_byte_addressable_store";

  pocl_cpuinfo_detect_device_info(device);
  pocl_topology_detect_device_info(device);

  if(!strcmp(device->llvm_cpu, "(unknown)"))
    device->llvm_cpu = NULL;

  // work-around LLVM bug where sizeof(long)=4
  #ifdef _CL_DISABLE_LONG
  device->has_64bit_long=0;
  #endif

}

void
pocl_hpx_uninit (cl_device_id device)
{
  struct data *d = (struct data*)device->data;
#ifdef CUSTOM_BUFFER_ALLOCATOR
  memory_region_t *region, *temp;
  DL_FOREACH_SAFE(d->mem_regions->mem_regions, region, temp)
    {
      DL_DELETE(d->mem_regions->mem_regions, region);
      free ((void*)region->chunks->start_address);
      free (region);    
    }
  d->mem_regions->mem_regions = NULL;
#endif  
  free (d);
  device->data = NULL;
}


#ifdef CUSTOM_BUFFER_ALLOCATOR
static int
allocate_aligned_buffer (struct data* d, void **memptr, size_t alignment, size_t size) 
{
  BA_LOCK(d->mem_regions->mem_regions_lock);
  chunk_info_t *chunk = alloc_buffer (d->mem_regions->mem_regions, size);
  if (chunk == NULL)
    {
      memory_region_t *new_mem_region = 
        (memory_region_t*)malloc (sizeof (memory_region_t));

      if (new_mem_region == NULL) 
        {
          BA_UNLOCK (d->mem_regions->mem_regions_lock);
          return ENOMEM;
        }

      /* Fallback to the minimum size in case of overflow. 
         Allocate a larger chunk to avoid allocation overheads
         later on. */
      size_t region_size = 
        std::max(std::min(size + ADDITIONAL_ALLOCATION_MAX_MB * 1024 * 1024, 
                size * ALLOCATION_MULTIPLE), size);

      assert (region_size >= size);

      void* space = NULL;
      if ((posix_memalign (&space, alignment, region_size)) != 0)
        {
          /* Failed to allocate a large region. Fall back to allocating 
             the smallest possible region for the buffer. */
          if ((posix_memalign (&space, alignment, size)) != 0) 
            {
              BA_UNLOCK (d->mem_regions->mem_regions_lock);
              return ENOMEM;
            }
          region_size = size;
        }

      init_mem_region (new_mem_region, (memory_address_t)space, region_size);
      new_mem_region->alignment = alignment;
      DL_APPEND (d->mem_regions->mem_regions, new_mem_region);
      chunk = alloc_buffer_from_region (new_mem_region, size);

      if (chunk == NULL)
      {
        printf("pocl error: could not allocate a buffer of size %zu from the newly created region of size %zu.\n",
               size, region_size);
        print_chunks(new_mem_region->chunks);
        /* In case the malloc didn't fail it should have been able to allocate 
           the buffer to a newly created Region. */
        assert (chunk != NULL);
      }
    }
  BA_UNLOCK (d->mem_regions->mem_regions_lock);
  
  *memptr = (void*) chunk->start_address;
  return 0;
}

#else

static int
allocate_aligned_buffer (struct data* d, void **memptr, size_t alignment, size_t size) 
{
  return posix_memalign (memptr, alignment, size);
}

#endif

void *
pocl_hpx_malloc (void *device_data, cl_mem_flags flags, size_t size, void *host_ptr)
{
  void *b;
  struct data* d = (struct data*)device_data;

  if (flags & CL_MEM_COPY_HOST_PTR)
    {
      if (allocate_aligned_buffer (d, &b, MAX_EXTENDED_ALIGNMENT, size) == 0)
        {
          memcpy (b, host_ptr, size);
          return b;
        }
      
      return NULL;
    }
  
  if (flags & CL_MEM_USE_HOST_PTR && host_ptr != NULL)
    {
      return host_ptr;
    }

  if (allocate_aligned_buffer (d, &b, MAX_EXTENDED_ALIGNMENT, size) == 0)
    return b;
  
  return NULL;
}

cl_int
pocl_hpx_alloc_mem_obj (cl_device_id device, cl_mem mem_obj)
{
  void *b = NULL;
  struct data* d = (struct data*)device->data;
  cl_int flags = mem_obj->flags;

  /* if memory for this global memory is not yet allocated -> do it */
  if (mem_obj->device_ptrs[device->global_mem_id].mem_ptr == NULL)
    {
      if (flags & CL_MEM_USE_HOST_PTR && mem_obj->mem_host_ptr != NULL)
        {
          b = mem_obj->mem_host_ptr;
        }
      else if (allocate_aligned_buffer (d, &b, MAX_EXTENDED_ALIGNMENT, 
                                        mem_obj->size) != 0)
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;

      if (flags & CL_MEM_COPY_HOST_PTR)
        memcpy (b, mem_obj->mem_host_ptr, mem_obj->size);
    
      mem_obj->device_ptrs[device->global_mem_id].mem_ptr = b;
      mem_obj->device_ptrs[device->global_mem_id].global_mem_id = 
        device->global_mem_id;
    }
  /* copy already allocated global mem info to devices own slot */
  mem_obj->device_ptrs[device->dev_id] = 
    mem_obj->device_ptrs[device->global_mem_id];
    
  return CL_SUCCESS;
}

#ifdef CUSTOM_BUFFER_ALLOCATOR
void
pocl_hpx_free (void *device_data, cl_mem_flags flags, void *ptr)
{
  struct data* d = (struct data*) device_data;
  memory_region_t *region = NULL;

  if (flags & CL_MEM_USE_HOST_PTR)
      return; /* The host code should free the host ptr. */

  region = free_buffer (d->mem_regions->mem_regions, (memory_address_t)ptr);

  assert(region != NULL && "Unable to find the region for chunk.");

#if FREE_EMPTY_REGIONS == 1
  BA_LOCK(d->mem_regions->mem_regions_lock);
  BA_LOCK(region->lock);
  if (region->last_chunk == region->chunks && 
      !region->chunks->is_allocated) 
    {
      /* All chunks have been deallocated. free() the whole 
         memory region at once. */
      DL_DELETE(d->mem_regions->mem_regions, region);
      free ((void*)region->last_chunk->start_address);
      free (region);    
    }  
  BA_UNLOCK(region->lock);
  BA_UNLOCK(d->mem_regions->mem_regions_lock);
#endif
}

#else

void
pocl_hpx_free (void *data, cl_mem_flags flags, void *ptr)
{
  if (flags & CL_MEM_COPY_HOST_PTR)
    return;
  
  free (ptr);
}
#endif

void
pocl_hpx_read (void *data, void *host_ptr, const void *device_ptr, size_t cb)
{
  if (host_ptr == device_ptr)
    return;

  memcpy (host_ptr, device_ptr, cb);
}

void
pocl_hpx_write (void *data, const void *host_ptr, void *device_ptr, size_t cb)
{
  if (host_ptr == device_ptr)
    return;
  
  memcpy (device_ptr, host_ptr, cb);
}


void
pocl_hpx_copy (void *data, const void *src_ptr, void *__restrict__ dst_ptr, size_t cb)
{
  if (src_ptr == dst_ptr)
    return;
  
  memcpy (dst_ptr, src_ptr, cb);
}

void
pocl_hpx_copy_rect (void *data,
                        const void *__restrict const src_ptr,
                        void *__restrict__ const dst_ptr,
                        const size_t *__restrict__ const src_origin,
                        const size_t *__restrict__ const dst_origin, 
                        const size_t *__restrict__ const region,
                        size_t const src_row_pitch,
                        size_t const src_slice_pitch,
                        size_t const dst_row_pitch,
                        size_t const dst_slice_pitch)
{
  char const *__restrict const adjusted_src_ptr = 
    (char const*)src_ptr +
    src_origin[0] + src_row_pitch * (src_origin[1] + src_slice_pitch * src_origin[2]);
  char *__restrict__ const adjusted_dst_ptr = 
    (char*)dst_ptr +
    dst_origin[0] + dst_row_pitch * (dst_origin[1] + dst_slice_pitch * dst_origin[2]);
  
  size_t j, k;

  /* TODO: handle overlaping regions */
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      memcpy (adjusted_dst_ptr + dst_row_pitch * j + dst_slice_pitch * k,
              adjusted_src_ptr + src_row_pitch * j + src_slice_pitch * k,
              region[0]);
}

void *
pocl_hpx_map_mem (void *data, void *buf_ptr, 
                      size_t offset, size_t size, void* host_ptr) 
{
  /* All global pointers of the hpx/CPU device are in 
     the host address space already, and up to date. */     
  return (char*)buf_ptr + offset;
}

#define FALLBACK_MAX_THREAD_COUNT 8
//#define DEBUG_MT
//#define DEBUG_MAX_THREAD_COUNT
/**
 * Return an estimate for the maximum thread count that should produce
 * the maximum parallelism without extra threading overheads.
 */
static
int 
get_max_thread_count(cl_device_id device) 
{
  /* if return THREAD_COUNT_ENV if set, 
     else return fallback or max_compute_units */
         printf("max_compute_units: %d\n", device->max_compute_units);
  if (device->max_compute_units == 0)
    return pocl_get_int_option (THREAD_COUNT_ENV, FALLBACK_MAX_THREAD_COUNT);
  else
    return pocl_get_int_option(THREAD_COUNT_ENV, device->max_compute_units * 20);
}

void
pocl_hpx_run
(void* data,
 _cl_command_node* cmd)
{
    unsigned int device = 0;
    cl_device_id device_ptr;
    cl_kernel kernel = cmd->command.run.kernel;
    size_t num_hpx_workers = hpx::get_os_thread_count();

    // initialize shared arrays
    std::vector<pocl_context *> pc_local(num_hpx_workers);
    std::vector<void**> wg_arguments(num_hpx_workers);
    std::vector<char> initialized(num_hpx_workers);
    for(size_t i = 0; i < num_hpx_workers; i++)
    {
        pc_local[i] = NULL;
        wg_arguments[i] = NULL;
        initialized[i] = 0;
    }

    /* Find which device number within the context corresponds
       to current device. */
    for (int i = 0; i < kernel->context->num_devices; ++i)
    {
        if (kernel->context->devices[i]->data == data)
        {
            device = i;
            device_ptr = kernel->context->devices[i];
            break;
        }
    }
 
    // initialize thread_arguments 
    struct thread_arguments ta;
    ta.data = data;
    ta.device = device;
    ta.pc_global = &cmd->command.run.pc;
    ta.workgroup = cmd->command.run.wg;
    ta.kernel_args = cmd->command.run.arguments;
    ta.kernel = cmd->command.run.kernel;
    ta.pc_local = pc_local.data();
    ta.wg_arguments = wg_arguments.data();
    ta.initialized = initialized.data();

    // run kernels
    /*
    hpx_foreach(workgroup_thread, &ta,
                                  ta.pc_global->num_groups[0],
                                  ta.pc_global->num_groups[1],
                                  ta.pc_global->num_groups[2]); 
    */
    size_t dim_x = ta.pc_global->num_groups[0];
    size_t dim_y = ta.pc_global->num_groups[1];
    size_t dim_z = ta.pc_global->num_groups[2];
    for(size_t z = 0; z < dim_z; z++)
    {
        for(size_t y = 0; y < dim_y; y++)
        {
            hpx::parallel::for_each(hpx::parallel::par(num_hpx_workers * 10),
                                    boost::counting_iterator<size_t>(0),
                                    boost::counting_iterator<size_t>(dim_x),
            [&y, &z, &ta] (size_t x)
            {
                workgroup_thread(&ta, x, y, z);
            });
        }
    }

 
    // cleanup thread_arguments
    for(size_t i = 0; i < num_hpx_workers; i++)
    {
        // cleanup only if thread actually initialized it
        if(ta.initialized[i])
        {
            // TODO cleanup locally created workgroup arguments and give them
            // to the memory manager
            for (int j = 0; j < kernel->num_args; ++j)
            {
                if (kernel->arg_info[j].is_local ||
                    kernel->arg_info[j].type == POCL_ARG_TYPE_IMAGE ||
                    kernel->arg_info[j].type == POCL_ARG_TYPE_SAMPLER)
                {
                    pocl_hpx_free (ta.data, 0, *(void **)(ta.wg_arguments[i][j]));
                }
            }
            for (int j = kernel->num_args;
                 j < kernel->num_args + kernel->num_locals;
                 ++j)
            {
                pocl_hpx_free (ta.data, 0, *(void **)(ta.wg_arguments[i][j]));
            }
             
            // free the allocated void* array
            free(ta.wg_arguments[i]); 

            // cleanup the local copy of pocl_context
            free(ta.pc_local[i]);
        }
    }
}

void workgroup_thread (void* p, size_t gid_x, size_t gid_y, size_t gid_z)
{
    // parse inputs
    struct thread_arguments *ta = (struct thread_arguments *) p;

    // get the cpu core id
    size_t hpx_worker_id = hpx::get_worker_thread_num();

    // initialize arguments if necessary.
    // every core has its own local copy of the input arguments
    // to avoid crossing numa-domains.
    if(ta->initialized[hpx_worker_id] != 1)
    {
        // set initialized flag
        ta->initialized[hpx_worker_id] = 1;

        // get number of thread arguments
        size_t arguments_len = ta->kernel->num_args + ta->kernel->num_locals;

        // allocate local worker struct
        ta->wg_arguments[hpx_worker_id] = (void**) malloc( 2 * arguments_len
                                                             * sizeof(void*) );

        // todo: initialize local thread arguments  
        {
            void** arguments = ta->wg_arguments[hpx_worker_id];
            void** arguments_ind = arguments + arguments_len;
            cl_kernel kernel = ta->kernel;
            struct pocl_argument *al;  
            for (int i = 0; i < kernel->num_args; ++i)
              {
                al = &(ta->kernel_args[i]);
                if (kernel->arg_info[i].is_local)
                  {
                    arguments[i] = &arguments_ind[i];
                    *(void **)(arguments[i]) = pocl_hpx_malloc(ta->data, 0, al->size, NULL);
                  }
                else if (kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER)
                {
                  /* It's legal to pass a NULL pointer to clSetKernelArguments. In 
                     that case we must pass the same NULL forward to the kernel.
                     Otherwise, the user must have created a buffer with per device
                     pointers stored in the cl_mem. */
                  if (al->value == NULL) 
                    {
                      arguments[i] = &arguments_ind[i];
                      *(void **)arguments[i] = NULL;
                    }
                  else
                    {
                      arguments[i] = 
                        &((*(cl_mem *)(al->value))->device_ptrs[ta->device].mem_ptr);
                    }
                }
                else if (kernel->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
                  {
                    dev_image_t di;
                    fill_dev_image_t(&di, al, ta->device);
                    void* devptr = pocl_hpx_malloc(ta->data, 0, sizeof(dev_image_t), NULL);
                    arguments[i] = &arguments_ind[i];
                    *(void **)(arguments[i]) = devptr;       
                    pocl_hpx_write (ta->data, &di, devptr, sizeof(dev_image_t));
                  }
                else if (kernel->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
                  {
                    dev_sampler_t ds;
                    
                    arguments[i] = &arguments_ind[i];
                    *(void **)(arguments[i]) = pocl_hpx_malloc
                      (ta->data, 0, sizeof(dev_sampler_t), NULL);
                    pocl_hpx_write (ta->data, &ds, *(void**)arguments[i],
                                        sizeof(dev_sampler_t));
                  }
                else
                  arguments[i] = al->value;
              }
          
            /* Allocate the automatic local buffers which are implemented as implicit
               extra arguments at the end of the kernel argument list. */
            for (int i = kernel->num_args;
                 i < kernel->num_args + kernel->num_locals;
                 ++i)
              {
                al = &(ta->kernel_args[i]);
                arguments[i] = &arguments_ind[i];
                *(void **)(arguments[i]) = pocl_hpx_malloc (ta->data, 0, al->size,
                                                                NULL);
              }
        }
        
        // initialize pocl context
        ta->pc_local[hpx_worker_id] = (struct pocl_context *)
                                            malloc(sizeof(struct pocl_context));
        *(ta->pc_local[hpx_worker_id]) = *(ta->pc_global);

    } 

    // set current group id
    struct pocl_context *pc = ta->pc_local[hpx_worker_id];
    pc->group_id[0] = gid_x;
    pc->group_id[1] = gid_y;
    pc->group_id[2] = gid_z;

    // do work.
    // syntax: workgroup(void**, pocl_context*)
    ta->workgroup(ta->wg_arguments[hpx_worker_id], pc);

}


