
#ifndef HPX_C_SPINLOCK_H
#define HPX_C_SPINLOCK_H

#if defined(__cplusplus)
extern "C"
{
#endif
 
    struct hpx_c_spinlock
    {
        long v_;
    };
     
    void hpx_c_spinlock_lock(struct hpx_c_spinlock* sp);
    void hpx_c_spinlock_unlock(struct hpx_c_spinlock* sp);
    void hpx_c_spinlock_init(struct hpx_c_spinlock* sp, void const*);
    void hpx_c_spinlock_destroy(struct hpx_c_spinlock* sp);
 
#if defined(__cplusplus)
}
#endif

#endif
