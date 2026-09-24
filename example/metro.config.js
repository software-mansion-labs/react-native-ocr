// Learn more https://docs.expo.io/guides/customizing-metro
const path = require('path');
const { getDefaultConfig } = require('expo/metro-config');

const root = path.resolve(__dirname, '..');
const pkg = require('../package.json');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

const { transformer, resolver } = config;

config.transformer = {
  ...transformer,
  babelTransformerPath: require.resolve('react-native-svg-transformer/expo'),
};
config.resolver = {
  ...resolver,
  assetExts: resolver.assetExts.filter((ext) => ext !== 'svg'),
  sourceExts: [...resolver.sourceExts, 'svg'],
};

config.resolver.assetExts.push('pte');

// The library is the workspace root and Yarn creates no node_modules link for
// it, so resolve imports of the package name straight to its TypeScript
// source. Everything the source imports resolves from the root node_modules
// by Metro's normal upward lookup.
if (!config.watchFolders.includes(root)) {
  config.watchFolders = [...config.watchFolders, root];
}
const defaultResolveRequest = config.resolver.resolveRequest;
config.resolver.resolveRequest = (context, moduleName, platform) => {
  if (moduleName === pkg.name) {
    return { type: 'sourceFile', filePath: path.join(root, 'src', 'index.ts') };
  }
  return (defaultResolveRequest ?? context.resolveRequest)(context, moduleName, platform);
};

module.exports = config;
