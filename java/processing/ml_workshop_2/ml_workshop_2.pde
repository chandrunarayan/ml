import Jama.*;
NeuralNetwork myNN;

void setup() {
  // create my neural network
  myNN = new NeuralNetwork(3, 3, 3, 0.1);
  
// Feed Forward  
  
  //input layer calculations
  double[][] awIH = {{0.9, 0.3, 0.4}, 
                     {0.2, 0.8, 0.2}, 
                     {0.1, 0.5, 0.6}};
  Matrix wIH = new Matrix(awIH);
  
  double[][] ainp = {{0.9}, 
                     {0.1}, 
                     {0.8}};
  Matrix inp = new Matrix(ainp);
  
  wIH.print(3,3);    
  inp.print(3,3);
  
  //hidden layer calculations
  Matrix hid_inp = wIH.times(inp);
  
  hid_inp.print(3,3);  
  hid_inp.transpose().print(3,3);
  
  Matrix hid_outp = Activator.sigmoid(hid_inp);
  
  hid_outp.print(3,3);  
  hid_outp.transpose().print(3,3);

  //output layer calculations
  double[][] awOH = {{0.3, 0.7, 0.5}, 
                     {0.6, 0.5, 0.2}, 
                     {0.8, 0.1, 0.9}};
  Matrix wOH = new Matrix(awOH);
  
  wOH.print(3,3); 
  hid_outp.print(3,3);
  
  //output of hidden layer is same as input of output layer
  Matrix out_outp = wOH.times(hid_outp);
  
  out_outp.print(3,3);  
  out_outp.transpose().print(3,3);  
  
  Matrix outp = Activator.sigmoid(out_outp);  
  
  outp.print(3,3);  
  outp.transpose().print(3,3);   
  
// Reverse Propagation    
  
}

void draw() {
}
