/* hpx.h - hpx device declarations.

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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

#ifndef POCL_HPX_EXT_H
#define POCL_HPX_EXT_H


#ifdef __cplusplus
extern "C" {
#endif

    // generic HPX
    size_t hpx_get_num_workers();
    size_t hpx_get_worker_id();
    void hpx_foreach(void (*)(void*, size_t, size_t, size_t),
                              void*, size_t, size_t, size_t);
    
    // threading
    typedef void* hpx_thread_t;
    int hpx_thread_create(hpx_thread_t*, void*(*)(void*), void* arg);
    int hpx_threads_join(hpx_thread_t*, size_t num_threads);

    ////////////////////////////
    /// MEMPOOL
    ///

    // adds a buffer to the mempool
    void mempool_release_clmem(void* mem, size_t size);

    // takes a buffer from the mempool.
    // returns NULL if no buffer of that size exists.
    void* mempool_get_clmem(size_t size);

    // "invalidates" all buffers, meaning, they do not have the content we need.
    void mempool_invalidade_all_buffers();
    // checks if a buffer is "invalidated".
    void mempool_is_buffer_valid(void*);

    // removes all buffers from the mempool and returns them in an array
    void** mempool_take_and_remove_all_buffers();

#ifdef __cplusplus
}
#endif

#endif /* POCL_HPX_EXT_H */
