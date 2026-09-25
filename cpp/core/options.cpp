#include "options.h"

#include <cmath>

#include "error.h"

namespace rnocr::core::options {

jsi::Object requireObject(jsi::Runtime &rt, const jsi::Value &value, const std::string &what) {
    if (!value.isObject() || value.asObject(rt).isArray(rt)) {
        throw error::InvalidArgument(what + ": expected an options object");
    }
    return value.asObject(rt);
}

namespace {
std::string path(const std::string &what, const char *name) {
    return what + "." + name;
}

double numberOf(jsi::Runtime &rt, const jsi::Value &value, const std::string &where) {
    if (!value.isNumber()) {
        throw error::InvalidArgument(where + ": expected a number");
    }
    const double number = value.asNumber();
    if (!std::isfinite(number)) {
        throw error::InvalidArgument(where + ": expected a finite number");
    }
    (void)rt;
    return number;
}
} // namespace

double requireNumber(jsi::Runtime &rt, const jsi::Object &object, const char *name, const std::string &what) {
    const jsi::Value value = object.getProperty(rt, name);
    if (value.isUndefined()) {
        throw error::InvalidArgument(path(what, name) + ": is required");
    }
    return numberOf(rt, value, path(what, name));
}

bool requireBool(jsi::Runtime &rt, const jsi::Object &object, const char *name, const std::string &what) {
    const jsi::Value value = object.getProperty(rt, name);
    if (!value.isBool()) {
        throw error::InvalidArgument(path(what, name) + ": expected a boolean");
    }
    return value.getBool();
}

std::optional<double> optionalNumber(jsi::Runtime &rt, const jsi::Object &object, const char *name, const std::string &what) {
    const jsi::Value value = object.getProperty(rt, name);
    if (value.isUndefined()) {
        return std::nullopt;
    }
    return numberOf(rt, value, path(what, name));
}

std::optional<std::string> optionalString(jsi::Runtime &rt, const jsi::Object &object, const char *name, const std::string &what) {
    const jsi::Value value = object.getProperty(rt, name);
    if (value.isUndefined()) {
        return std::nullopt;
    }
    if (!value.isString()) {
        throw error::InvalidArgument(path(what, name) + ": expected a string");
    }
    return value.getString(rt).utf8(rt);
}

} // namespace rnocr::core::options
