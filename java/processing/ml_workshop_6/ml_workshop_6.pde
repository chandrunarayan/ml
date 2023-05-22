/**
 * Bush School CPJava Class Final Project
 * Project Details: https://chandrunarayan.github.io/cpjava/final_projects/
 * 1. Build a complete Java Neural Network from scratch
 * 2. Test the Neural Network using 2 scenarios
 *    a. Predict equation of line using a supplied set of points
 *    b. Classify hand written 28x28 pixel numerals from 0-9
 * Adapted for Bush School by Chandru Narayan
 * from "Make your own Neural Network" by Tariq Rashid
 */

// Import NIST Java Matrix Library
// https://math.nist.gov/javanumerics/jama/Jama-1.0.3.jarhttps://math.nist.gov/javanumerics/jama/Jama-1.0.3.jar
// https://math.nist.gov/javanumerics/jama/doc/

import Jama.*;

// Globals
NeuralNetwork bushNN;
int input_nodes = 3;
int hidden_nodes = 3;
int output_nodes = 3;
float learning_rate = 0.1;
int nInp = 10000;

boolean debug = false;

// Main
void setup() {
  // create my neural network
  bushNN = new NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate);

  // Create randomized nInp input training data in a loop as a Column Matrix
  // carefully note how the column matrix is initialized
  // This will eventualy come from a file of input values

  for (int i = 0; i < nInp; i++) {
    Matrix inp = MatrixUtil.mcrud(3, 1);  // input data column matrix
    MatrixUtil.mprint(debug, "printing input to neural network", inp);

    // Create a tgt matrix where each elemt is 3 times original
    // first get the 2D array inside matrix
    //  clone it
    double [][] outA = inp.getArrayCopy();
    // scalar multiplication by 0.5
    for (int p = 0; p < outA.length; p++) {
      for (int q = 0; q < outA[p].length; q++) {
        outA[p][q] *= 0.5;
      }
    }
    
    Matrix tgt = new Matrix(outA);

    // Train the neural network for a given set of training inputs for which answer is known!
    // This is accomplished through backward propagation
    bushNN.train(inp, tgt);
  }


  // Create unseen random input data as a Column Matrix
  Matrix inp = MatrixUtil.mcrud(3, 1);  // input data column matrix
  MatrixUtil.mprint(true, "printing unseen random input to neural network", inp);

  Matrix [] out_new = bushNN.predict(inp);

  MatrixUtil.mprint(debug, "printing hidden layer inputs: weighted sum", out_new[0]);
  MatrixUtil.mprint(debug, "printing hidden layer outputs: sigmoid(weighted sum)", out_new[1]);
  MatrixUtil.mprint(debug, "printing output layer inputs = hidden_layer outputs", out_new[1]);
  MatrixUtil.mprint(true, "printing output layer outputs = final outputs", out_new[2]);
}

void draw() {
}
