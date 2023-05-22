static class Activator {
  static Matrix sigmoid(Matrix in) {
    double [][] inA = in.getArray();
    double [][] outA = in.getArrayCopy();
    for (int i = 0; i < inA.length; i++) {
      for (int j = 0; j < inA[i].length; j++) {
        outA[i][j] = 1.0/(1+Math.exp(-inA[i][j]));
      }
    }
    return new Matrix(outA);
  }
}
