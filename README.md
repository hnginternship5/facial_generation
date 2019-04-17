# ML facial Generation

This is a face generation application based on a race and gender supplied as input.

# Usage

# Virtual Environments

### ****Using Conda***

### Create an environment

```conda create --name nameofenvironment --file requirements.txt```

### Activate
```conda activate nameofenvironment```

### Run
```python cli.py age race gender```

# Issues

Mac Users

If you get the following error when running
```
OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized. 
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/. 
```

run ```conda install nomkl```