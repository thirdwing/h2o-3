package hex.deeplearning;

import water.Key;
import water.MRTask;
import water.fvec.*;
import water.parser.BufferedString;
import water.gpu.ImageIter;
import water.util.Log;


import java.io.IOException;
import java.util.ArrayList;

/**
 * DRemoteTask-based Deep Learning.
 * Every node has access to all the training data which leads to optimal CPU utilization and training accuracy IFF the data fits on every node.
 */
public class DeepLearningTask2 extends MRTask<DeepLearningTask2> {
  static {
    System.loadLibrary("cudart");
    System.loadLibrary("cublas");
    System.loadLibrary("curand");
    System.loadLibrary("Native");
  }
  /**
   * Construct a DeepLearningTask2 where every node trains on the entire training dataset
   * @param jobKey Job ID
   * @param train Frame containing training data
   * @param model_info Initial DeepLearningModelInfo (weights + biases)
   * @param sync_fraction Fraction of the training data to use for one SGD iteration
   */
  public DeepLearningTask2(Key jobKey, Frame train, DeepLearningModelInfo model_info, float sync_fraction, int iteration) {
    assert(sync_fraction > 0);
    _jobKey = jobKey;
    _fr = train;
    _sharedmodel = model_info;
    _sync_fraction = sync_fraction;
    _iteration = iteration;
  }

  /**
   * Returns the aggregated DeepLearning model that was trained by all nodes (over all the training data)
   * @return model_info object
   */
  public DeepLearningModelInfo model_info() { return _sharedmodel; }

  final private Key _jobKey;
  final private Frame _fr;
  private DeepLearningModelInfo _sharedmodel;
  final private float _sync_fraction;
  private DeepLearningTask _res;
  private final int _iteration;

  /**
   * Do the local computation: Perform one DeepLearningTask (with run_local=true) iteration.
   * Pass over all the data (will be replicated in dfork() here), and use _sync_fraction random rows.
   * This calls DeepLearningTask's reduce() between worker threads that update the same local model_info via Hogwild!
   * Once the computation is done, reduce() will be called
   */
  @Override
  public void setupLocal() {
    super.setupLocal();
    Vec response = _sharedmodel._train.vec(_sharedmodel._train.numCols() - 1);

    ArrayList<String> img_lst = new ArrayList<>();
    for (int i = 0; i < _sharedmodel._train.numCols() - 1; i++) {
      for (int j = 0; j < _sharedmodel._train.numRows(); j++) {
        BufferedString str = new BufferedString();
        img_lst.add(_sharedmodel._train.vec(i).atStr(str, j).toString());
      }
    }

    ArrayList<Float> label_lst = new ArrayList<>();
    for (int i = 0; i < response.length(); i++) {
      label_lst.add((float)response.at(i));
    }

    try {
      int batch_size = 40;

      ImageIter img_iter = new ImageIter(img_lst, label_lst, batch_size, "/tmp", 224, 224);
      img_iter.Reset();
      int iter = 0;
      while(img_iter.Nest()){
        float[] data = img_iter.getData();
        float[] labels = img_iter.getLabel();
        _sharedmodel._image_classify_gpu.train(data, labels, true);
        Log.info("Training epoch: " + iter);
        iter++;
      }

    }catch(IOException ie) {
      ie.printStackTrace();
    }

  }

  @Override
  public void map(Key key) {}

  @Override
  public void reduce(DeepLearningTask2 drt) {}

  /**
   * Finish up the work after all nodes have reduced their models via the above reduce() method.
   * All we do is average the models and add to the global training sample counter.
   * After this returns, model_info() can be queried for the updated model.
   */
  @Override
  protected void postGlobal() {
    super.postGlobal();
    _sharedmodel.add_processed_local(_sharedmodel._train.numRows());
  }
}
