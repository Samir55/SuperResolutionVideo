#pragma once

#include <iostream>
#include <vector>
#include <mutex>
#include <queue>
#include <condition_variable>

using namespace std;

template<typename T>
class concurrent_priority_queue {

public:
    /**
     * Pushes the given element in the vector.
     *
     * @param element   The element to be pushed.
     */
    void push(T element) {
        {
            lock_guard<mutex> guard(mtx);
            data.push(element);
        }

        cv.notify_all();
    }

    /**
     * Returns the first element in the underlying vector.
     * Note: blocks the execution in case the vector is empty, until some one pushes.
     *
     * @return  The oldest element in the vector.
     */
    T pop() {
        T element;

        {
            unique_lock<mutex> lk(mtx);

            while (empty()) {
                cv.wait(lk, [=] { return !empty(); });
            }

            element = data.top();
            data.pop();

            lk.unlock();
        }

        return element;
    }

    /**
     * Returns a copy of the first element in the underlying vector.
     * Note: blocks the execution in case the vector is empty, until some one pushes.
     *
     * @return  A copy of the oldest element in the vector.
     */
    T top() {
        T element;

        {
            unique_lock<mutex> lk(mtx);

            while (empty()) {
                cv.wait(lk, [=] { return !empty(); });
            }

            element = data.top();

            lk.unlock();
        }

        return element;
    }

    /**
     * Clears the data container.
     */
    void clear() {
        {
            lock_guard<mutex> guard(mtx);
            while (!data.empty())
                data.pop();
        }
    }

    /**
     * Returns the vector size.
     *
     * @return  The vector size.
     */
    size_t size() {
        // STL vector is thread-safe reading.
        return data.size();
    }

    /**
     * Checks if vector is empty.
     *
     * @return  @true if vector is empty, @false otherwise.
     */
    bool empty() {
        return size() == 0;
    }

private:
    priority_queue<T> data;
    mutex mtx;
    condition_variable cv;
};
