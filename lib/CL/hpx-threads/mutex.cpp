/* hpx-threads/mutex.cpp: mutex for hpx-threads

   Copyright (c) 2014 Martin Stumpf
   
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

#include "mutex.h"
#include <errno.h>
#include <iostream>

#include <hpx/lcos/local/spinlock.hpp>
#include <boost/atomic.hpp>

static boost::atomic<size_t> num_locks((size_t)0);

typedef hpx::lcos::local::spinlock lock_type;

int hpx_mutex_init(hpx_mutex_t* lock_raw)
{
    try {
        *lock_raw = (hpx_mutex_t) new lock_type();
    } catch (hpx::exception e) {
        std::cerr << "Error while initializing lock!" << std::endl;
        exit(1);
    }
    size_t current_num_locks = ++num_locks;
    std::cout << "mutex init!    (" << current_num_locks << " left)" << std::endl;
    return 0;   
}

int hpx_mutex_destroy(hpx_mutex_t* lock_raw)
{
    try {
        lock_type* lock = (lock_type*) *lock_raw;
        delete(lock);
        *lock_raw = NULL;
    } catch (hpx::exception e) {
        std::cerr << "Error while destroying lock!" << std::endl;
        exit(1);
    }
    size_t current_num_locks = --num_locks;

    std::cout << "mutex destroy! (" << current_num_locks << " left)" << std::endl;
    return 0;
}

int hpx_mutex_lock(hpx_mutex_t* lock_raw)
{
    try {
        lock_type* lock = (lock_type*) *lock_raw;
        lock->lock();
    } catch (hpx::exception e) {
        std::cerr << "Error while locking lock!" << std::endl;
        exit(1);
    }
    return 0;
}

int hpx_mutex_unlock(hpx_mutex_t* lock_raw)
{
    try {
        lock_type* lock = (lock_type*) *lock_raw;
        lock->unlock();
    } catch (hpx::exception e) {
        std::cerr << "Error while unlocking lock!" << std::endl;
        exit(1);
    }
    return 0;
}

