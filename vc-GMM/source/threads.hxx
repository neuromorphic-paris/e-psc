/******************************************************************************/
//
//	Copyright (C) 2018, Florian Hirschberger <florian.hirschberger@uol.de>
//
//	LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
//	ACADEMIC FREE LICENSE (AFL) v3.0.
//
/******************************************************************************/

#ifndef THREAD_POOL
#define THREAD_POOL

#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>

class tp {
    std::vector<std::thread> threads;
    size_t nthreads;
    size_t cap;
    size_t up;
    size_t ret;
    std::mutex m;
    std::mutex dyn;
    std::condition_variable c;
    std::condition_variable w;

    void (tp::*call)(size_t t);
    const void *args;

    template <class L>
    void wrap(size_t t);

    void loop(void);
    void wait();

public:
    tp(size_t nthreads);
    ~tp();

    template <class L>
    void parallel(size_t tasks, const L& immu);

    size_t size(void);
};

tp::tp(size_t nthreads)
    : threads()
    , nthreads(nthreads)
    , cap(0)
    , ret(0)
{
    for (size_t t = 0; t < nthreads; t++) {
        threads.push_back(std::thread(&tp::loop, this));
    }
    wait();
}

tp::~tp()
{
    call = nullptr;
    args = nullptr;

    c.notify_all();
    for (size_t t = 0; t < nthreads; t++) {
        threads[t].join();
    }
}

void
tp::loop(void)
{
    size_t t;
    {
        std::lock_guard<std::mutex> lock(m);
        t = cap++;
    }

    while (true)
    {
        {
            std::unique_lock<std::mutex> lock(m);
            ret++;
            if (ret == nthreads) {
                w.notify_all();
            }
            c.wait(lock);
        }

        if (this->call) {
            (this->*call)(t);
        } else {
            break;
        }
    }
}

void
tp::wait()
{
    std::unique_lock<std::mutex> lock(m);
    while (ret < nthreads) {
        w.wait(lock);
    }
}

template <class L>
void
tp::wrap(size_t t)
{
    size_t delta = (cap + nthreads - 1) / nthreads;
    size_t from  = t * delta;
    size_t to    = (t + 1) * delta;

    for (size_t it = from; it < std::min(cap, to); it++) {
        (*reinterpret_cast<const L*>(args))(it, t);
    }
}


template <class L>
void
tp::parallel(size_t tasks, const L& immu)
{
    call = &tp::wrap<L>;
    args = &immu;

    cap = tasks;
    ret = 0;
    up  = 0;

    c.notify_all();
    wait();
}

size_t
tp::size(void)
{
    return nthreads;
}

#endif
