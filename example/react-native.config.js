const path = require('path');
const pkg = require('../package.json');

// The library is the workspace root, and Yarn creates no node_modules link for
// the root workspace, so autolinking cannot resolve it by name. Point it at the
// directory explicitly (the create-react-native-library convention).
module.exports = {
  project: {
    ios: {
      automaticPodsInstallation: true,
    },
  },
  dependencies: {
    [pkg.name]: {
      root: path.join(__dirname, '..'),
    },
  },
};
