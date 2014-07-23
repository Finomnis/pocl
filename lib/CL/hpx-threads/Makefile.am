# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/lib/CL/hpx_threads.
# 
# Copyright (c) 2011-2012 Universidad Rey Juan Carlos and 
#                         Tampere University of Technology
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#lib_LTLIBRARIES = libhpxthread.la
bin_PROGRAMS = hpxtest


#libhpxthread_la_SOURCES = test.cpp	
hpxtest_SOURCES = test.cpp

hpxtest_CPPFLAGS = @HPX_INCLUDE_FLAGS@
hpxtest_LDFLAGS = @HPX_LINK_FLAGS@ @HPX_LIBS@

#libpocl_la_CPPFLAGS = -I$(top_srcdir)/fix-include -I$(top_srcdir)/fix-include/OpenCL -I$(top_srcdir)/include -I$(top_srcdir)/lib/CL/devices $(OCL_ICD_CFLAGS)
#libpocl_la_LDFLAGS = -lltdl @PTHREAD_CFLAGS@ -version-info ${LIB_VERSION}
#libpocl_la_LDFLAGS += -Wl,-z,defs 

#libpocl_la_LIBADD = devices/libpocl-devices.la
#libpocl_la_LIBADD += ${LTDL_LIBS}

#Kludge: compile pocl_llvm_api.cc into a library of its own.
#The source file is necessarely a C++ file, and having a C++ file
#causes the entire pocl project to be linked with g++, which in turn
#exposes a gcc/autoconf m4 macro issue
# see: http://stackoverflow.com/questions/17055279/autotools-for-pthreads-not-setting-correct-linker-flags
#noinst_LTLIBRARIES = libpoclllvm.la

#CLEANFILES = hpxtest
