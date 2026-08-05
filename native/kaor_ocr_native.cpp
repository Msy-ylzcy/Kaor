#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <opencv2/core.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

class BufferView {
public:
    explicit BufferView(PyObject* object) {
        if (PyObject_GetBuffer(
                object,
                &view_,
                PyBUF_ND | PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
            throw std::runtime_error("expected an object exposing a strided buffer");
        }
        acquired_ = true;
        if (view_.ndim != 2 || view_.itemsize != 1 || view_.shape == nullptr ||
            view_.strides == nullptr || view_.strides[1] != 1 ||
            view_.strides[0] < view_.shape[1]) {
            PyBuffer_Release(&view_);
            acquired_ = false;
            throw std::runtime_error("expected a two-dimensional uint8 image");
        }
        if (view_.format != nullptr && std::string(view_.format) != "B") {
            PyBuffer_Release(&view_);
            acquired_ = false;
            throw std::runtime_error("expected unsigned uint8 pixels");
        }
    }

    BufferView(const BufferView&) = delete;
    BufferView& operator=(const BufferView&) = delete;

    ~BufferView() {
        if (acquired_) {
            PyBuffer_Release(&view_);
        }
    }

    int rows() const { return static_cast<int>(view_.shape[0]); }
    int columns() const { return static_cast<int>(view_.shape[1]); }

    cv::Mat matrix() const {
        return cv::Mat(
            rows(),
            columns(),
            CV_8UC1,
            view_.buf,
            static_cast<std::size_t>(view_.strides[0]));
    }

private:
    Py_buffer view_{};
    bool acquired_ = false;
};

PyObject* changed_pixel_ratio(PyObject*, PyObject* arguments) {
    PyObject* previous_object = nullptr;
    PyObject* current_object = nullptr;
    int threshold = 0;
    if (!PyArg_ParseTuple(
            arguments,
            "OOi:changed_pixel_ratio",
            &previous_object,
            &current_object,
            &threshold)) {
        return nullptr;
    }

    try {
        BufferView previous(previous_object);
        BufferView current(current_object);
        if (previous.rows() != current.rows() ||
            previous.columns() != current.columns()) {
            throw std::runtime_error("frame signatures must have matching shapes");
        }
        const auto pixel_count =
            static_cast<double>(previous.rows()) * previous.columns();
        if (pixel_count <= 0.0) {
            throw std::runtime_error("frame signatures must not be empty");
        }
        if (threshold <= 0) {
            return PyFloat_FromDouble(1.0);
        }
        if (threshold > 255) {
            return PyFloat_FromDouble(0.0);
        }

        cv::Mat difference;
        cv::Mat changed;
        cv::absdiff(previous.matrix(), current.matrix(), difference);
        cv::compare(difference, cv::Scalar(threshold), changed, cv::CMP_GE);
        const auto ratio = cv::countNonZero(changed) / pixel_count;
        return PyFloat_FromDouble(ratio);
    } catch (const cv::Exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (const std::exception& error) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_BufferError, error.what());
        }
    }
    return nullptr;
}

PyMethodDef methods[] = {
    {
        "changed_pixel_ratio",
        changed_pixel_ratio,
        METH_VARARGS,
        "Return the fraction of uint8 pixels whose absolute delta meets a threshold.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_kaor_ocr_native",
    "Optional fused frame-change operations for Kaor OCR.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__kaor_ocr_native() {
    return PyModule_Create(&module);
}
