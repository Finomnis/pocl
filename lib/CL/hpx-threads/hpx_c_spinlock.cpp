/* hpx-threads/hpx_c_spinlock.cpp: mutex for hpx-threads

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

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/register_locks.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <boost/thread/locks.hpp>

#if defined(BOOST_WINDOWS)
#  include <boost/smart_ptr/detail/spinlock_w32.hpp>
#else
#  if !defined(__ANDROID__) && !defined(ANDROID)
#    include <boost/smart_ptr/detail/spinlock_sync.hpp>
#    if defined( __ia64__ ) && defined( __INTEL_COMPILER )
#      include <ia64intrin.h>
#    endif
#  endif
#endif

#include "hpx_c_spinlock.h"

static void hpx_c_spinlock_yield(std::size_t k)
{
    if (k < 4) //-V112
    {
    }
#if defined(BOOST_SMT_PAUSE)
    else if(k < 16)
    {
        BOOST_SMT_PAUSE
    }
#endif
    else if(k < 32 || k & 1) //-V112
    {
        if (hpx::threads::get_self_ptr())
        {
            hpx::this_thread::suspend(hpx::threads::pending,
                "spinlock::yield");
        }
        else
        {
#if defined(BOOST_WINDOWS)
            Sleep(0);
#elif defined(BOOST_HAS_PTHREADS)
            sched_yield();
#else
#endif
        }
    }
    else
    {
        if (hpx::threads::get_self_ptr())
        {
            hpx::this_thread::suspend(hpx::threads::pending,
                "local::spinlock::yield");
        }
        else
        {
#if defined(BOOST_WINDOWS)
            Sleep(1);
#elif defined(BOOST_HAS_PTHREADS)
            // g++ -Wextra warns on {} or {0}
            struct timespec rqtp = { 0, 0 };

            // POSIX says that timespec has tv_sec and tv_nsec
            // But it doesn't guarantee order or placement

            rqtp.tv_sec = 0;
            rqtp.tv_nsec = 1000;

            nanosleep( &rqtp, 0 );
#else
#endif
        }
    }
}

bool acquire_lock(hpx_c_spinlock* sp)
{
#if defined(BOOST_WINDOWS)
    long r = BOOST_INTERLOCKED_EXCHANGE(&sp->v_, 1);
    BOOST_COMPILER_FENCE
#else
    long r = __sync_lock_test_and_set(&sp->v_, 1);
#endif
    return r == 0;
}

void relinquish_lock(hpx_c_spinlock* sp)
{
#if defined(BOOST_WINDOWS)
    BOOST_COMPILER_FENCE
    *const_cast<long volatile*>(&sp->v_) = 0;
#else
    __sync_lock_release(&sp->v_);
#endif
}

void hpx_c_spinlock_lock(hpx_c_spinlock* sp)
{
    HPX_ITT_SYNC_PREPARE(sp);

    for (std::size_t k = 0; !acquire_lock(sp); ++k)
    {
        hpx_c_spinlock_yield(k);
    }

    HPX_ITT_SYNC_ACQUIRED(sp);
    hpx::util::register_lock(sp);
}

void hpx_c_spinlock_unlock(hpx_c_spinlock* sp)
{
    HPX_ITT_SYNC_RELEASING(sp);

    relinquish_lock(sp);

    HPX_ITT_SYNC_RELEASED(sp);
    hpx::util::unregister_lock(sp);
}

void hpx_c_spinlock_init(hpx_c_spinlock* sp, void const*)
{
    sp->v_ = 0;
}

void hpx_c_spinlock_destroy(hpx_c_spinlock* sp)
{
}


