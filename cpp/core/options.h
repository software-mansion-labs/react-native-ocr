#pragma once

#include <optional>
#include <string>

#include <jsi/jsi.h>

/**
 * Readers for the plain option objects some ops take as their last argument.
 * Every reader names the offending property in its InvalidArgument message.
 */
namespace rnocr::core::options {
namespace jsi = facebook::jsi;

/** Requires `value` to be a (non-array) object. */
jsi::Object requireObject(jsi::Runtime &rt, const jsi::Value &value, const std::string &what);

/** Reads a required finite number property. */
double requireNumber(jsi::Runtime &rt, const jsi::Object &object, const char *name, const std::string &what);

/** Reads a required boolean property. */
bool requireBool(jsi::Runtime &rt, const jsi::Object &object, const char *name, const std::string &what);

/** Reads an optional finite number property; `undefined` yields nullopt. */
std::optional<double> optionalNumber(jsi::Runtime &rt, const jsi::Object &object, const char *name, const std::string &what);

/** Reads an optional string property; `undefined` yields nullopt. */
std::optional<std::string> optionalString(jsi::Runtime &rt, const jsi::Object &object, const char *name, const std::string &what);

} // namespace rnocr::core::options
