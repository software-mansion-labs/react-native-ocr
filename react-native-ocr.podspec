require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

# package.json carries the npm-style "git+https://....git" form; CocoaPods wants
# a plain URL for the homepage and a bare git URL for the source.
repo_url = package["repository"]["url"].sub(/\Agit\+/, "").sub(/\.git\z/, "")

# OpenCV comes from the same CocoaPod react-native-executorch links, so the app
# holds exactly one opencv2. Its podspec picks the pod from the build config its
# postinstall writes (`opencvPod`), else from whether react-native-fast-opencv
# is installed next to it; mirror that choice here.
rne_pkg = `node --print "require.resolve('react-native-executorch/package.json', { paths: ['#{__dir__}'] })"`.strip
rne_dir = File.dirname(rne_pkg)
rne_config_path = File.join(rne_dir, "rne-build-config.json")
rne_config = File.exist?(rne_config_path) ? JSON.parse(File.read(rne_config_path)) : {}

if rne_config["enableOpencv"] == false
  raise(defined?(Pod::Informative) ? Pod::Informative : StandardError, <<~MESSAGE)
    react-native-ocr needs OpenCV, but react-native-executorch was installed
    without it. Add the "ocr" feature to the app's package.json:

      "react-native-executorch": { "features": ["ocr"] }

    then reinstall dependencies and run `pod install` again.
  MESSAGE
end

detected_opencv_pod =
  if Dir.exist?(File.join(rne_dir, "..", "react-native-fast-opencv"))
    "FastOpenCV-iOS"
  else
    "opencv-rne"
  end
opencv_pod = rne_config["opencvPod"] || detected_opencv_pod

Pod::Spec.new do |s|
  s.name         = "react-native-ocr"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  # Has to match react-native-executorch, which requires iOS 17.
  s.platforms    = { :ios => "17.0" }
  s.source       = { :git => "#{repo_url}.git", :tag => "v#{s.version}" }

  s.source_files = "ios/**/*.{h,mm}", "cpp/**/*.{h,cpp}"
  # Host GoogleTest suites, built by cpp/tests/CMakeLists.txt, never by the app.
  s.exclude_files = "cpp/tests/**/*"
  # Only the Objective-C entry point has to be visible to the app; the C++
  # headers are reached through HEADER_SEARCH_PATHS. Keeping the public set
  # small avoids header-name collisions under `use_frameworks!`.
  s.public_header_files = "ios/**/*.h"

  s.pod_target_xcconfig = {
    "USE_HEADERMAP" => "YES",
    "CLANG_CXX_LANGUAGE_STANDARD" => "c++20",
    "HEADER_SEARCH_PATHS" => "\"$(PODS_TARGET_SRCROOT)/cpp\"",
    "WARNING_CFLAGS" => "-Wno-documentation",
    # react-native-executorch ships no simulator x86_64 slice and drops the
    # arch; a dependent that keeps it has nothing to link against.
    "EXCLUDED_ARCHS[sdk=iphonesimulator*]" => "x86_64",
  }
  s.user_target_xcconfig = {
    "EXCLUDED_ARCHS[sdk=iphonesimulator*]" => "x86_64",
  }

  # opencv2.xcframework is a static binary, which a dynamic framework may not
  # carry; declaring the pod static keeps `use_frameworks!` apps installable.
  s.static_framework = true

  if opencv_pod == "opencv-rne"
    s.dependency "opencv-rne", "~> 4.11.0"
  else
    Pod::UI.puts "[react-native-ocr] using #{opencv_pod} for OpenCV, matching react-native-executorch" if defined?(Pod::UI)
    s.dependency opencv_pod
  end

  install_modules_dependencies(s)
end
