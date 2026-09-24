module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    // The worklets plugin lives here and not in the library's build: the
    // library ships `'worklet'` directives untouched and the consuming app's
    // Metro workletizes them, as react-native-executorch does. Babel applies
    // this root config to the library's `src/` as well, which Metro pulls in
    // through the resolver in metro.config.js.
    plugins: ['react-native-worklets/plugin'],
  };
};
