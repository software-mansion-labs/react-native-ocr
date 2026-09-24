package com.swmansion.rnocr

import com.facebook.react.BaseReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.model.ReactModuleInfo
import com.facebook.react.module.model.ReactModuleInfoProvider

class RnOcrPackage : BaseReactPackage() {

  override fun getModule(name: String, reactContext: ReactApplicationContext): NativeModule? =
    when (name) {
      RnOcrModule.NAME -> RnOcrModule(reactContext)
      else -> null
    }

  override fun getReactModuleInfoProvider(): ReactModuleInfoProvider =
    ReactModuleInfoProvider {
      mapOf(
        RnOcrModule.NAME to ReactModuleInfo(
          RnOcrModule.NAME,
          RnOcrModule.NAME,
          false,
          false,
          true,
          false,
          true
        )
      )
    }
}
