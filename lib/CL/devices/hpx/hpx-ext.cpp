/* OpenCL hpx device implementation.

   Copyright (c)      2014 Martin Stumpf
   
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

#include <hpx/config.hpp>
#include <hpx/hpx.hpp>

#include "hpx-ext.h"

struct hpx_thread{
public:
    hpx_thread(hpx::future<void> && future_) : future(std::move(future_)){}
    hpx::future<void> future;
};

int hpx_thread_create(hpx_thread_t* out, void*(*f)(void*), void* arg)
{
    try{

        // run thread asynchronously, store future in stack
        hpx_thread* thread = new hpx_thread(hpx::async(f, arg));
        
        // return reference to future
        *out = (hpx_thread_t) thread;

    } catch (...) {
        std::cerr << "Error creating threads!" << std::endl
                  << "Make sure that your executable runs withing an HPX context."
                  << std::endl;
        return 1;
    }
    
    return 0;
}

int hpx_threads_join(hpx_thread_t* threads, size_t num_threads)
{

    try{

        // create a vector of futures
        std::vector<hpx::future<void>> futures(num_threads);
        for(size_t i = 0; i < num_threads; i++)
        {
            hpx_thread* thread = (hpx_thread*)(threads[i]);
            futures[i] = std::move(thread->future);
            delete(thread);
        }
    
        // wait for all futures to finish
        hpx::when_all(futures).wait();

    } catch (...) {
        std::cerr << "Error joining threads!" << std::endl;
        return 1;
    }

    return 0;
}

size_t hpx_get_num_workers()
{

    return hpx::get_os_thread_count();
}

size_t hpx_get_worker_id()
{

    return hpx::get_worker_thread_num();
}

void hpx_foreach(void (*)(void*, size_t, size_t, size_t),
                          void*, size_t, size_t, size_t)
{

}







