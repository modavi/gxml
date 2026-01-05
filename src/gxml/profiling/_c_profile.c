/*
 * _c_profile.c - Ultra-fast profiling C extension for GXML
 * 
 * This module provides a minimal C implementation of the profiling hot path:
 *   - push_marker(marker_id) - Record push event with timestamp
 *   - pop_marker(marker_id) - Record pop event with timestamp
 *   - get_events() - Return collected events to Python
 *   - clear_events() - Reset the event buffer
 *   - ProfiledFunction - Callable wrapper type for profiled functions
 * 
 * The goal is to minimize per-marker overhead by avoiding Python object
 * creation in the hot path. Events are stored in a pre-allocated C array
 * and only converted to Python objects when get_events() is called.
 * 
 * Build with: pip install -e .
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <time.h>
#endif

/* ============================================================================
 * Platform-Specific High-Resolution Timer
 * ============================================================================ */

#ifdef _WIN32
static LARGE_INTEGER qpc_frequency;
static int qpc_initialized = 0;

static inline double get_time(void) {
    LARGE_INTEGER counter;
    if (!qpc_initialized) {
        QueryPerformanceFrequency(&qpc_frequency);
        qpc_initialized = 1;
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / qpc_frequency.QuadPart;
}
#else
static inline double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/* ============================================================================
 * Event Storage
 * ============================================================================ */

/* Event structure: marker_id (positive=push, negative=pop) and timestamp */
typedef struct {
    int marker_id;
    double timestamp;
} ProfileEvent;

/* Pre-allocated event buffer */
#define INITIAL_CAPACITY 65536
#define MAX_CAPACITY (1024 * 1024 * 16)  /* 16M events max */

static ProfileEvent* events = NULL;
static size_t event_count = 0;
static size_t event_capacity = 0;

/* Initialize or grow the event buffer */
static int ensure_capacity(void) {
    if (events == NULL) {
        events = (ProfileEvent*)malloc(INITIAL_CAPACITY * sizeof(ProfileEvent));
        if (events == NULL) return -1;
        event_capacity = INITIAL_CAPACITY;
        return 0;
    }
    
    if (event_count >= event_capacity) {
        size_t new_capacity = event_capacity * 2;
        if (new_capacity > MAX_CAPACITY) {
            PyErr_SetString(PyExc_MemoryError, "Profiler event buffer exceeded maximum size");
            return -1;
        }
        ProfileEvent* new_events = (ProfileEvent*)realloc(events, new_capacity * sizeof(ProfileEvent));
        if (new_events == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to grow profiler event buffer");
            return -1;
        }
        events = new_events;
        event_capacity = new_capacity;
    }
    return 0;
}

/* Inline push/pop for use within C (no Python overhead) */
static inline int c_push_marker(int marker_id) {
    if (ensure_capacity() < 0) return -1;
    events[event_count].marker_id = marker_id;
    events[event_count].timestamp = get_time();
    event_count++;
    return 0;
}

static inline int c_pop_marker(int marker_id) {
    if (ensure_capacity() < 0) return -1;
    events[event_count].marker_id = -(marker_id + 1);
    events[event_count].timestamp = get_time();
    event_count++;
    return 0;
}

/* ============================================================================
 * ProfiledFunction - Callable wrapper type
 * ============================================================================
 * 
 * A C-level callable that wraps a Python function with profiling.
 * When called, it does: push_marker -> call func -> pop_marker
 * all within C, avoiding the Python->C boundary crossing for push/pop.
 */

typedef struct {
    PyObject_HEAD
    PyObject* func;         /* The wrapped Python callable */
    int marker_id;          /* Pre-registered marker ID */
    PyObject* func_name;    /* Cached __name__ attribute */
    PyObject* func_module;  /* Cached __module__ attribute */
    PyObject* func_doc;     /* Cached __doc__ attribute */
    PyObject* func_dict;    /* __dict__ for the wrapper */
} ProfiledFunctionObject;

static void
ProfiledFunction_dealloc(ProfiledFunctionObject* self) {
    Py_XDECREF(self->func);
    Py_XDECREF(self->func_name);
    Py_XDECREF(self->func_module);
    Py_XDECREF(self->func_doc);
    Py_XDECREF(self->func_dict);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
ProfiledFunction_call(ProfiledFunctionObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* result;
    
    /* Push marker (direct C call, no boundary crossing) */
    if (c_push_marker(self->marker_id) < 0) {
        return NULL;
    }
    
    /* Call the wrapped function */
    result = PyObject_Call(self->func, args, kwargs);
    
    /* Pop marker (direct C call, no boundary crossing) */
    /* Note: We pop even if the function raised an exception */
    if (c_pop_marker(self->marker_id) < 0) {
        Py_XDECREF(result);
        return NULL;
    }
    
    return result;
}

/* Descriptor protocol - allows the wrapper to work as a method */
static PyObject*
ProfiledFunction_descr_get(ProfiledFunctionObject* self, PyObject* obj, PyObject* type) {
    if (obj == NULL || obj == Py_None) {
        Py_INCREF(self);
        return (PyObject*)self;
    }
    /* Return a bound method */
    return PyMethod_New((PyObject*)self, obj);
}

/* Attribute access for functools.wraps compatibility */
static PyObject*
ProfiledFunction_getattro(ProfiledFunctionObject* self, PyObject* name) {
    const char* name_str = PyUnicode_AsUTF8(name);
    if (name_str == NULL) return NULL;
    
    /* Handle special attributes */
    if (strcmp(name_str, "__wrapped__") == 0) {
        Py_INCREF(self->func);
        return self->func;
    }
    if (strcmp(name_str, "__name__") == 0 && self->func_name) {
        Py_INCREF(self->func_name);
        return self->func_name;
    }
    if (strcmp(name_str, "__module__") == 0 && self->func_module) {
        Py_INCREF(self->func_module);
        return self->func_module;
    }
    if (strcmp(name_str, "__doc__") == 0 && self->func_doc) {
        Py_INCREF(self->func_doc);
        return self->func_doc;
    }
    if (strcmp(name_str, "__dict__") == 0) {
        if (self->func_dict == NULL) {
            self->func_dict = PyDict_New();
            if (self->func_dict == NULL) return NULL;
        }
        Py_INCREF(self->func_dict);
        return self->func_dict;
    }
    
    /* Fall back to default attribute lookup */
    return PyObject_GenericGetAttr((PyObject*)self, name);
}

static PyTypeObject ProfiledFunctionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_c_profile.ProfiledFunction",
    .tp_doc = "C-level profiled function wrapper",
    .tp_basicsize = sizeof(ProfiledFunctionObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_dealloc = (destructor)ProfiledFunction_dealloc,
    .tp_call = (ternaryfunc)ProfiledFunction_call,
    .tp_descr_get = (descrgetfunc)ProfiledFunction_descr_get,
    .tp_getattro = (getattrofunc)ProfiledFunction_getattro,
};

/*
 * create_profiled_function(func, marker_id) -> ProfiledFunction
 * 
 * Create a C-level wrapper that profiles the given function.
 */
static PyObject* create_profiled_function(PyObject* self, PyObject* args) {
    PyObject* func;
    int marker_id;
    
    if (!PyArg_ParseTuple(args, "Oi", &func, &marker_id)) {
        return NULL;
    }
    
    if (!PyCallable_Check(func)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be callable");
        return NULL;
    }
    
    ProfiledFunctionObject* wrapper = PyObject_New(ProfiledFunctionObject, &ProfiledFunctionType);
    if (wrapper == NULL) return NULL;
    
    Py_INCREF(func);
    wrapper->func = func;
    wrapper->marker_id = marker_id;
    wrapper->func_dict = NULL;
    
    /* Cache function attributes for functools.wraps-like behavior */
    wrapper->func_name = PyObject_GetAttrString(func, "__name__");
    if (wrapper->func_name == NULL) PyErr_Clear();
    
    wrapper->func_module = PyObject_GetAttrString(func, "__module__");
    if (wrapper->func_module == NULL) PyErr_Clear();
    
    wrapper->func_doc = PyObject_GetAttrString(func, "__doc__");
    if (wrapper->func_doc == NULL) PyErr_Clear();
    
    return (PyObject*)wrapper;
}

/* ============================================================================
 * PerfMarker - C-level context manager
 * ============================================================================
 * 
 * A C-level context manager for performance marking.
 * Implements __enter__ and __exit__ entirely in C for minimal overhead.
 * 
 * Usage from Python:
 *     with PerfMarker(marker_id) as m:
 *         # ... code to measure ...
 */

typedef struct {
    PyObject_HEAD
    int marker_id;
} PerfMarkerObject;

static void
PerfMarker_dealloc(PerfMarkerObject* self) {
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int
PerfMarker_init(PerfMarkerObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"marker_id", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &self->marker_id)) {
        return -1;
    }
    return 0;
}

static PyObject*
PerfMarker_enter(PerfMarkerObject* self, PyObject* Py_UNUSED(ignored)) {
    /* Record push event directly - no function call overhead */
    if (c_push_marker(self->marker_id) < 0) {
        return NULL;
    }
    Py_INCREF(self);
    return (PyObject*)self;
}

static PyObject*
PerfMarker_exit(PerfMarkerObject* self, PyObject* args) {
    /* Record pop event directly - no function call overhead */
    /* We ignore the exception arguments (exc_type, exc_val, exc_tb) */
    if (c_pop_marker(self->marker_id) < 0) {
        return NULL;
    }
    Py_RETURN_FALSE;  /* Don't suppress exceptions */
}

static PyMethodDef PerfMarker_methods[] = {
    {"__enter__", (PyCFunction)PerfMarker_enter, METH_NOARGS, "Enter the context manager."},
    {"__exit__", (PyCFunction)PerfMarker_exit, METH_VARARGS, "Exit the context manager."},
    {NULL}
};

static PyTypeObject PerfMarkerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_c_profile.PerfMarker",
    .tp_doc = "C-level context manager for performance marking",
    .tp_basicsize = sizeof(PerfMarkerObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)PerfMarker_init,
    .tp_dealloc = (destructor)PerfMarker_dealloc,
    .tp_methods = PerfMarker_methods,
};

/*
 * create_perf_marker(marker_id) -> PerfMarker
 * 
 * Create a C-level context manager for the given marker ID.
 * This is faster than calling PerfMarker(marker_id) from Python
 * because it avoids the __init__ call overhead.
 */
static PyObject* create_perf_marker(PyObject* self, PyObject* args) {
    int marker_id;
    
    if (!PyArg_ParseTuple(args, "i", &marker_id)) {
        return NULL;
    }
    
    PerfMarkerObject* marker = PyObject_New(PerfMarkerObject, &PerfMarkerType);
    if (marker == NULL) return NULL;
    
    marker->marker_id = marker_id;
    return (PyObject*)marker;
}

/* ============================================================================
 * Python Interface Functions
 * ============================================================================ */

/*
 * push_marker(marker_id: int) -> None
 * 
 * Record a push event (function entry) with the current timestamp.
 * marker_id should be a pre-registered integer ID.
 */
static PyObject* push_marker(PyObject* self, PyObject* args) {
    int marker_id;
    
    if (!PyArg_ParseTuple(args, "i", &marker_id)) {
        return NULL;
    }
    
    if (ensure_capacity() < 0) {
        return NULL;
    }
    
    events[event_count].marker_id = marker_id;
    events[event_count].timestamp = get_time();
    event_count++;
    
    Py_RETURN_NONE;
}

/*
 * pop_marker(marker_id: int) -> None
 * 
 * Record a pop event (function exit) with the current timestamp.
 * Uses negative marker_id encoding: -(marker_id + 1)
 */
static PyObject* pop_marker(PyObject* self, PyObject* args) {
    int marker_id;
    
    if (!PyArg_ParseTuple(args, "i", &marker_id)) {
        return NULL;
    }
    
    if (ensure_capacity() < 0) {
        return NULL;
    }
    
    /* Encode pop as negative: -(marker_id + 1) */
    events[event_count].marker_id = -(marker_id + 1);
    events[event_count].timestamp = get_time();
    event_count++;
    
    Py_RETURN_NONE;
}

/*
 * get_events() -> List[Tuple[int, float]]
 * 
 * Return all collected events as a list of (marker_id, timestamp) tuples.
 * This is the only place where Python objects are created.
 */
static PyObject* get_events(PyObject* self, PyObject* args) {
    PyObject* list = PyList_New(event_count);
    if (list == NULL) return NULL;
    
    for (size_t i = 0; i < event_count; i++) {
        PyObject* tuple = Py_BuildValue("(id)", events[i].marker_id, events[i].timestamp);
        if (tuple == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, tuple);
    }
    
    return list;
}

/*
 * clear_events() -> None
 * 
 * Reset the event buffer without deallocating memory.
 */
static PyObject* clear_events(PyObject* self, PyObject* args) {
    event_count = 0;
    Py_RETURN_NONE;
}

/*
 * get_event_count() -> int
 * 
 * Return the current number of events.
 */
static PyObject* get_event_count(PyObject* self, PyObject* args) {
    return PyLong_FromSize_t(event_count);
}

/* ============================================================================
 * Module Definition
 * ============================================================================ */

static PyMethodDef ProfilerMethods[] = {
    {"push_marker", push_marker, METH_VARARGS, 
     "Record a push (function entry) event with timestamp."},
    {"pop_marker", pop_marker, METH_VARARGS,
     "Record a pop (function exit) event with timestamp."},
    {"get_events", get_events, METH_NOARGS,
     "Return all collected events as list of (marker_id, timestamp) tuples."},
    {"clear_events", clear_events, METH_NOARGS,
     "Clear the event buffer."},
    {"get_event_count", get_event_count, METH_NOARGS,
     "Return the current event count."},
    {"create_profiled_function", create_profiled_function, METH_VARARGS,
     "Create a C-level profiled function wrapper."},
    {"create_perf_marker", create_perf_marker, METH_VARARGS,
     "Create a C-level context manager for performance marking."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef profilemodule = {
    PyModuleDef_HEAD_INIT,
    "_c_profile",
    "High-performance profiling C extension for GXML",
    -1,
    ProfilerMethods
};

PyMODINIT_FUNC PyInit__c_profile(void) {
    PyObject* module;
    
    /* Initialize the ProfiledFunction type */
    if (PyType_Ready(&ProfiledFunctionType) < 0) {
        return NULL;
    }
    
    /* Initialize the PerfMarker type */
    if (PyType_Ready(&PerfMarkerType) < 0) {
        return NULL;
    }
    
    module = PyModule_Create(&profilemodule);
    if (module == NULL) {
        return NULL;
    }
    
    /* Add ProfiledFunction type to module */
    Py_INCREF(&ProfiledFunctionType);
    if (PyModule_AddObject(module, "ProfiledFunction", (PyObject*)&ProfiledFunctionType) < 0) {
        Py_DECREF(&ProfiledFunctionType);
        Py_DECREF(module);
        return NULL;
    }
    
    /* Add PerfMarker type to module */
    Py_INCREF(&PerfMarkerType);
    if (PyModule_AddObject(module, "PerfMarker", (PyObject*)&PerfMarkerType) < 0) {
        Py_DECREF(&PerfMarkerType);
        Py_DECREF(module);
        return NULL;
    }
    
    return module;
}
