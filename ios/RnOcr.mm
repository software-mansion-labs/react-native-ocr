#import "RnOcr.h"
#import "../cpp/RnOcr.h"
#import <React/RCTBridge+Private.h>
#import <jsi/jsi.h>

@implementation RnOcr {
    __weak RCTBridge *_bridge;
}

@synthesize bridge = _bridge;

RCT_EXPORT_MODULE()

- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params {
    return std::make_shared<facebook::react::NativeRnOcrSpecJSI>(params);
}

RCT_EXPORT_SYNCHRONOUS_TYPED_METHOD(NSNumber *, install) {
    RCTCxxBridge *cxxBridge = (RCTCxxBridge *)self.bridge;

    if (cxxBridge == nil)
        return @NO;

    facebook::jsi::Runtime *jsiRuntime = (facebook::jsi::Runtime *)cxxBridge.runtime;

    if (jsiRuntime == nil)
        return @NO;

    rnocr::install(*jsiRuntime);

    return @YES;
}

@end
