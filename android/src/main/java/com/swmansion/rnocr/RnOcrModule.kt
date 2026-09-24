package com.swmansion.rnocr

import com.facebook.react.bridge.JavaScriptContextHolder
import com.facebook.react.bridge.ReactApplicationContext

class RnOcrModule(reactContext: ReactApplicationContext) : NativeRnOcrSpec(reactContext) {

  override fun getName(): String = NAME

  override fun install(): Boolean {
    val contextHolder: JavaScriptContextHolder =
      reactApplicationContext.javaScriptContextHolder ?: return false
    nativeInstall(contextHolder.get())
    return true
  }

  companion object {
    const val NAME = "RnOcr"

    init {
      System.loadLibrary("RnOcr")
    }

    @JvmStatic
    external fun nativeInstall(jsi: Long)
  }
}
