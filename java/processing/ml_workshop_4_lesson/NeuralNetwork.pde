/**
 * Bush School CPJava Class Final Project
 * Project Details: https://chandrunarayan.github.io/cpjava/final_projects/
 * NeuralNetwork class with predict() & train() functions 
 */
class NeuralNetwork {
  /** number of input nodes */
  int iNodes;
  /** number of hidden nodes */
  int hNodes;  
  /** number of output nodes */
  int oNodes;
  /** learning rate */
  float lRate;
  /** weights matrix in between input and hidden layers */
  Matrix wIH;  
  /** weights matrix in between hidden and output layers */
  Matrix wHO; 
  
  /**
   * Constructor for the Neural Network Class.
   * Initialize properties
   */  
  /** initialize nodes and lr */
  NeuralNetwork(int iNodes_, int hNodes_, int oNodes_, float lRate_) {
    iNodes = ??;
    hNodes = ??;
    oNodes = ??;
    lRate = ??;

    // setup weights
    
    // initial input_hidden weights
    // this will eventually be created using a normal distribution of random numbers
    double[][] awIH = {{??, ??, ??},
      {??, ??, ??},
      {??, ??, ??}};
    wIH = new Matrix(??);  // weights matrix in between input and hidden layers
    MatrixPrint.mprint("printing initial input_hidden weights", ??);

    // initial hidden_output weights
    // this will eventually be created using a normal distribution of random numbers
    double[][] awHO = {{??, ??, ??},
      {??, ??, ??},
      {??, ??, ??}};
    wHO = new Matrix(??);  // weights matrix in between hidden and output layers
    MatrixPrint.mprint("printing initial hidden_output weights", ??);

  }
  
  /**
   * predict() function implementing feed forward calculations.
   * @param inp: Matrix of input values to Neural Network
   * @return out: Matrix of calculated output values from the Neural Network
   */  
  Matrix predict(Matrix inp) {
    
    // hidden layer calculations
    // hidden layer inputs: weighted sum
    Matrix hid_inp = wIH.??(??);  // dot product to create the weighted sum
    MatrixPrint.mprint("printing hidden layer inputs: weighted sum", ??);

    // hidden layer outputs: sigmoid(weighted sum)
    // note: output of hidden layer is same as input of output layer
    Matrix hid_outp = ??.??(??);  // sigmoid activation of the weighted sum see Activator class
    MatrixPrint.mprint("printing hidden layer outputs: sigmoid(weighted sum)", ??);

    //output layer calculations
    //output of hidden layer is same as input of output layer
    Matrix out_outp = wHO.??(??);  // dot product to create the weighted sum
    MatrixPrint.mprint("printing output layer inputs: weighted sum", ??);

    // calculate sigmoid activation of the weighted sum of the output layer
    Matrix out = ??.??(??); // sigmoid activation of the weighted sum see Activator class
    return out;  // return the sigmoid activation of the weighted sum
  }

  /**
   * train() function for implementing backward propagation.
   * @param inp: Matrix of input values to train the Neural Network
   */ 
  void train(Matrix inp) {
    // Reverse Propagation
    // first feed forward!
    Matrix out = this.predict(??);
    MatrixPrint.mprint("printing first output from neural network before back propagation", ??);
  }
}
