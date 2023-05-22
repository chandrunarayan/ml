import Jama.*;
NeuralNetwork myNN;
int input_nodes = 3;
int hidden_nodes = 3;
int output_nodes = 3;
float learning_rate = 0.1;

void setup() {
  // create my neural network
  myNN = new NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate);

  double[][] ainp = {{0.9},
    {0.1},
    {0.8}};
    
  Matrix inp = new Matrix(ainp);
  System.out.println("printing inputs to neural network");  
  inp.print(input_nodes, input_nodes);
  //inp.transpose().print(input_nodes, input_nodes);
  
  Matrix out = myNN.predict(inp);
  System.out.println("printing final output from neural network");  
  out.print(output_nodes, output_nodes);
  //out.transpose().print(input_nodes, input_nodes);

  myNN.train();
}

void draw() {
}
