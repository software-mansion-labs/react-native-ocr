module.exports = {
  dependency: {
    platforms: {
      android: {
        sourceDir: './android',
        packageImportPath: 'import com.swmansion.rnocr.RnOcrPackage;',
        packageInstance: 'new RnOcrPackage()',
      },
      ios: {},
    },
  },
};
