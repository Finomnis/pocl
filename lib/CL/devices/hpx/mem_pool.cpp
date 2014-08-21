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

#include "mem_pool.hpp"

extern "C" {
    #include "config.h"
}

#include <cstdlib>
#include <cassert>

typedef std::stack<void*, std::vector<void*> >      mem_list;
typedef std::map<std::size_t, mem_list>::iterator   pool_iterator;

// NOT THREAD SAFE! needs external synchronization.
void*
mem_pool::allocate(std::size_t size)
{
    // try to find already allocated memory
    pool_iterator it = pool.find(size);
    if(it != pool.end())
    {
        if(! it->second.empty())
        {
            void* buf = it->second.top();
            it->second.pop();
            return buf;
        }
    }

    // allocate new iterator
    void* new_buf;
    
    int ret_val = posix_memalign(&new_buf, MAX_EXTENDED_ALIGNMENT, size);
    assert(ret_val == 0);

    // remember size of the allocated memory
    mem_sizes[new_buf] = size;

    return new_buf;
}

void
mem_pool::release(void* mem)
{
    // get size of memory
    std::map<void*, std::size_t>::iterator mem_size = mem_sizes.find(mem);
    assert(mem_size != mem_sizes.end());

    std::size_t size = mem_size->second;

    // add to pool
    pool[size].push(mem);
}
