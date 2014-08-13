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

#include "nd_range_iterator.hpp"

#include <boost/assert.hpp>

nd_pos::nd_pos(size_t x_,
               size_t y_,
               size_t z_,
               size_t size_x_,
               size_t size_y_,
               size_t size_z_)
    : x(x_), y(y_), z(z_), size_x(size_x_), size_y(size_y_), size_z(size_z_),
      size_total(size_x_ * size_y_ * size_z_)
{}

nd_range_iterator
nd_range_iterator::begin(size_t size_x, size_t size_y, size_t size_z)
{
    return nd_range_iterator(nd_pos(0, 0, 0, size_x, size_y, size_z));
}

nd_range_iterator
nd_range_iterator::end(size_t size_x, size_t size_y, size_t size_z)
{
    return nd_range_iterator(nd_pos(0, 0, size_z, size_x, size_y, size_z));
}

nd_range_iterator::nd_range_iterator()
    : pos(0,0,0,0,0,0)
{}

nd_range_iterator::nd_range_iterator(value_type& pos_)
    : pos(pos_)
{
    BOOST_ASSERT((
                    (pos.x < pos.size_x) &&
                    (pos.y < pos.size_y) &&
                    (pos.z < pos.size_z)
                 ) || (
                    (pos.x == 0) &&
                    (pos.y == 0) && 
                    (pos.z == pos.size_z)
                 ));
}

nd_range_iterator::nd_range_iterator(value_type&& pos_)
    : pos(std::forward<value_type>(pos_))
{
    BOOST_ASSERT((
                    (pos.x < pos.size_x) &&
                    (pos.y < pos.size_y) &&
                    (pos.z < pos.size_z)
                 ) || (
                    (pos.x == 0) &&
                    (pos.y == 0) && 
                    (pos.z == pos.size_z)
                 ));
}

void
nd_range_iterator::increment()
{
    if(pos.z == pos.size_z)
        return;

    pos.x++;

    if(pos.x == pos.size_x)
    {
        pos.x = 0;
        pos.y++;
        
        if(pos.y == pos.size_y)
        {
            pos.y = 0;
            pos.z++;
        }
    }

    BOOST_ASSERT((
                    (pos.x < pos.size_x) &&
                    (pos.y < pos.size_y) &&
                    (pos.z < pos.size_z)
                 ) || (
                    (pos.x == 0) &&
                    (pos.y == 0) && 
                    (pos.z == pos.size_z)
                 ));
}

void
nd_range_iterator::decrement()
{
    if(pos.x == 0)
    {
        if(pos.y == 0)
        {
            if(pos.z == 0)
            {
            }
            else
            {
                pos.z--;
                pos.y = pos.size_y - 1;
                pos.x = pos.size_x - 1;
            }
        }
        else
        {
            pos.y--;
            pos.x = pos.size_x - 1;
        }
    }
    else
    {
        pos.x--;
    }

    BOOST_ASSERT((
                    (pos.x < pos.size_x) &&
                    (pos.y < pos.size_y) &&
                    (pos.z < pos.size_z)
                 ) || (
                    (pos.x == 0) &&
                    (pos.y == 0) && 
                    (pos.z == pos.size_z)
                 ));
}

bool
nd_range_iterator::equal(nd_range_iterator const& other) const
{
    BOOST_ASSERT((other.pos.size_x == pos.size_x) &&
                 (other.pos.size_y == pos.size_y) && 
                 (other.pos.size_z == pos.size_z));

    return (other.pos.x == pos.x) &&
           (other.pos.y == pos.y) &&
           (other.pos.z == pos.z);
}

nd_range_iterator::reference
nd_range_iterator::dereference() const
{
    return pos;
}

void
nd_range_iterator::advance(nd_range_iterator::difference_type diff)
{
    size_t pos_abs = (pos.z * pos.size_y + pos.y) * pos.size_x + pos.x;
    
    if(diff < 0)
    {
        size_t to_subtract = (size_t)(-diff);
        if(to_subtract >= pos_abs)
        {
            pos.x = 0;
            pos.y = 0;
            pos.z = 0;
        }
        else
        {
            pos_abs -= to_subtract;
            pos.x = pos_abs % pos.size_x;
            pos_abs /= pos.size_x;
            pos.y = pos_abs % pos.size_y;
            pos_abs /= pos.size_y;
            pos.z = pos_abs;
        }
    }
    else
    {
        size_t to_add = (size_t) diff;
        if(to_add + pos_abs >= pos.size_total)
        {
            pos.x = 0;
            pos.y = 0;
            pos.z = pos.size_z;
        }
        else
        {
            pos_abs += to_add;        
            pos.x = pos_abs % pos.size_x;
            pos_abs /= pos.size_x;
            pos.y = pos_abs % pos.size_y;
            pos_abs /= pos.size_y;
            pos.z = pos_abs;
        }
    }

    BOOST_ASSERT((
                    (pos.x < pos.size_x) &&
                    (pos.y < pos.size_y) &&
                    (pos.z < pos.size_z)
                 ) || (
                    (pos.x == 0) &&
                    (pos.y == 0) && 
                    (pos.z == pos.size_z)
                 ));
}

nd_range_iterator::difference_type
nd_range_iterator::distance_to(nd_range_iterator const& other) const
{
    size_t pos_abs = (pos.z * pos.size_y + pos.y) * pos.size_x + pos.x;
    size_t pos_abs_other = (other.pos.z * other.pos.size_y + other.pos.y)
                           * other.pos.size_x + other.pos.x;

    if(pos_abs_other >= pos_abs)
    {
        return (difference_type) (pos_abs_other - pos_abs);
    }
    else
    {
        return - (difference_type) (pos_abs - pos_abs_other);
    }
}






