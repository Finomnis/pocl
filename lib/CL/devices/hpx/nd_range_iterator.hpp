/* hpx.h - hpx device declarations.

   Copyright (c) 2014 Martin Stumpf, Ste||ar
   
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

#ifndef POCL_HPX_ND_RANGE_ITERATOR_HPP
#define POCL_HPX_ND_RANGE_ITERATOR_HPP

#include <boost/iterator/iterator_facade.hpp>

struct nd_pos
{
    public:
        nd_pos(size_t, size_t, size_t, size_t, size_t, size_t);
        size_t x;
        size_t y;
        size_t z;
        const size_t size_x;
        const size_t size_y;
        const size_t size_z;
        const size_t size_total;
};

class nd_range_iterator
  : public boost::iterator_facade < nd_range_iterator,
                                    nd_pos const,
                                    std::random_access_iterator_tag >
{
    public:
        nd_range_iterator();
        nd_range_iterator(value_type&);
        nd_range_iterator(value_type&&);

        static nd_range_iterator begin(size_t, size_t, size_t);
        static nd_range_iterator end(size_t, size_t, size_t);

    private:
        friend class boost::iterator_core_access;
        
        void increment();

        void decrement();

        bool equal(nd_range_iterator const& other) const;

        reference dereference() const;

        void advance(difference_type);

        difference_type
        distance_to(nd_range_iterator const& other) const;


    private:
        nd_pos pos;
};









#endif
