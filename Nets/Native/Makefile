#  -----------------------------------------------------------------------------------------------------------------------
#  Instructions:
#
#  1.  Activate your Tensorflow virtualenv before running this script. Before running this script, 'python' command in the
#      terminal should refer to the Python interpreter associated to your Tensorflow installation.
#
#  2.  Run 'make', it should produce a new file named 'high_dim_filter.so'.
#
#  3.  If this script fails, please refer to https://www.tensorflow.org/extend/adding_an_op#build_the_op_library for help.
#
#  -----------------------------------------------------------------------------------------------------------------------

# Define the compiler
CC := g++

#Use GPU Implementation?
USE_GPU := 1

# Read Tensorflow paths
TF_INC := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') 

# Is the Tensorflow version >= 1.4?
TF_VERSION_GTE_1_4 := $(shell expr `python -c 'import tensorflow as tf; print(tf.__version__)' | cut -f1,2 -d.` \>= 1.4)

# Flags required for all cases
CFLAGS := -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -shared -fPIC -I$(TF_INC) -O2 

# Set a special flag if we are on macOS
ifeq ($(shell uname -s), Darwin)
	CFLAGS += -undefined dynamic_lookup
endif

# Set some more flags if the Tensorflow version is >= 1.4
ifeq ($(TF_VERSION_GTE_1_4), 1)
    CFLAGS += -I$(TF_INC)/external/nsync/public 
	LDFLAGS := -L$(TF_LIB) -ltensorflow_framework -L/usr/local/cuda/lib64 -lcuda -lcudart
else
	LDFLAGS :=
endif

# Define build targets
.PHONY: all clean

all: shift_corr.so

shift_corr.so: shift_corr.cc shift_corr.cu.o
	g++ $(CFLAGS) -o shift_corr.so shift_corr.cc shift_corr.cu.o $(LDFLAGS) -D  RUN_GPU=$(USE_GPU)

shift_corr.cu.o: shift_corr.cu.cc
	nvcc -std=c++11 -c -o shift_corr.cu.o shift_corr.cu.cc $(TF_CFLAGS) -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -D GOOGLE_CUDA=1

clean:
	$(RM) shift_corr.so shift_corr.cu.o