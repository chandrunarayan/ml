class NeuralNetwork {
  int iNodes;
  int hNodes;
  int oNodes;
  float lRate;
  Matrix wIH;
  Matrix inp;
  Matrix outp;
  Matrix wHO;

  NeuralNetwork(int iNodes_, int hNodes_, int oNodes_, float lRate_) {
    // initialize nodes and lr
    iNodes = iNodes_;
    hNodes = hNodes_;
    oNodes = oNodes_;
    lRate = lRate_;
    
    // setup weights
    double[][] awIH = {{0.9, 0.3, 0.4},
      {0.2, 0.8, 0.2},
      {0.1, 0.5, 0.6}};
    System.out.println("printing initial input_hidden weights");      
    wIH = new Matrix(awIH);
    wIH.print(hidden_nodes, hidden_nodes);
    
    double[][] awHO = {{0.3, 0.7, 0.5},
      {0.6, 0.5, 0.2},
      {0.8, 0.1, 0.9}};
    System.out.println("printing initial hidden_output weights");      
    wHO = new Matrix(awHO);    
    wHO.print(hidden_nodes, hidden_nodes);
  }

  void train() {
    // Reverse Propagation
  }

  Matrix predict(Matrix inp) {
    //hidden layer calculations
    Matrix hid_inp = wIH.times(inp);
    
    System.out.println("printing hidden layer inputs: weighted sum");
    hid_inp.print(hidden_nodes, hidden_nodes);
    //hid_inp.transpose().print(hidden_nodes, hidden_nodes);

    Matrix hid_outp = Activator.sigmoid(hid_inp);

    System.out.println("printing hidden layer outputs: sigmoid(weighted sum)");
    System.out.println("this is also the input to the output layer");
    hid_outp.print(hidden_nodes, hidden_nodes);
    //hid_outp.transpose().print(hidden_nodes, hidden_nodes);

    //output layer calculations
    //output of hidden layer is same as input of output layer
    Matrix out_outp = wHO.times(hid_outp);

    System.out.println("printing output layer inputs: weighted sum");
    out_outp.print(hidden_nodes, hidden_nodes);
    //out_outp.transpose().print(hidden_nodes, hidden_nodes);

   return Activator.sigmoid(out_outp);
  }
}
