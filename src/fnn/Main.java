package fnn;

public class Main {

  public static void main(String[] args) {
    
    // Copy data from file, shuffle them and write them in 2D array
    float[][] data = DataLib.copyDataToArray("heart_disease_dataset.data" /*filename*/, "," /*separator*/);
    
    //DataLib.printData();
    
    NeuralNet nn = new NeuralNet(data, 100 /*batchSize*/, 1 /*nb classes*/,5 /*nb neurons in h1 */, false /* has 1 hidden layer */);

    nn.train(100/*nb of epochs*/);
    nn.train(200/*nb of epochs*/);
    nn.train(500/*nb of epochs*/);
  }

}
