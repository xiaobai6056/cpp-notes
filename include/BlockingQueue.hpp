#pragma once
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <utility>
using namespace std;

template <typename T>
class BlockingQueue
{
private:

	BlockingQueue(const BlockingQueue& rhs);
	BlockingQueue& operator =(const BlockingQueue& rhs);
	mutable mutex _mutex;
	mutable condition_variable _condvar;
	deque<T> _queue;
	bool isShutDown = false;
public:
	BlockingQueue()
		: _mutex()
		  , _condvar()
		  , _queue()
	{
	}
	~BlockingQueue()
	{
		//cout << "blocking queue release" << endl;
	}
	void ShutDown()
	{
		lock_guard<mutex> lock(_mutex);
		isShutDown = true;
		_condvar.notify_all();
		_queue.clear();
	}
	bool IsShutDown()
	{
		lock_guard<mutex> lock(_mutex);
		return isShutDown;
	}
	void Put(T task)
	{
		lock_guard<mutex> lock(_mutex);
		if (!isShutDown)
		{   if(_queue.size()>=1)//只保留最新的10帧
				_queue.pop_front();
			_queue.push_back(std::move(task));
			_condvar.notify_one();
		}
	}

	T Take()
	{
		unique_lock<mutex> lock(_mutex);
		_condvar.wait(lock, [this] { return isShutDown || !_queue.empty(); });
		if (isShutDown || _queue.empty())
		{
			throw exception();
		}
		T front(std::move(_queue.front()));
		_queue.pop_front();
		return front;
	}

	/// Get a copy of the first element without popping it.
	/// Blocks until an element is available or the queue is shut down.
	T front() const
	{
		unique_lock<mutex> lock(_mutex);
		_condvar.wait(lock, [this] { return isShutDown || !_queue.empty(); });
		if (isShutDown || _queue.empty())
		{
			throw exception();
		}
		return _queue.front();
	}

	/// Create a snapshot copy of the queue (including its current elements).
	BlockingQueue<T> clone() const
	{
		unique_lock<mutex> lock(_mutex);
		BlockingQueue<T> copy;
		copy._queue = _queue;
		copy.isShutDown = isShutDown;
		return copy;
	}

	size_t Size() const
	{
		lock_guard<mutex> lock(_mutex);
		return _queue.size();
	}
};
