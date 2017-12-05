using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace convnet
{
    class Net
    {
        public List<layer> layers;
        public Net()
        {
            layers = new List<layer>();
        }

        public void makeLayers(List<String> defs)
        {

                // few checks
                //assert(defs.length >= 2, 'Error! At least one input layer and one loss layer are required.');
                //assert(defs[0].type == 'input', 'Error! First layer must be the input layer, to declare size of inputs');

                // desugar layer_defs for adding activation, dropout layers etc
                
                List<String> new_defs = new List<String>();
                for (var i = 0; i < defs.Count; i++)
                {
                    
                    Layer_def def = JsonConvert.DeserializeObject<Layer_def>(defs[i]);

                    if (def.type == "softmax" || def.type == "svm")
                        {
                            // add an fc layer here, there is no reason the user should
                            // have to worry about this and we almost always want to
                            new_defs.Add("{ type:  'fc', num_neurons:"+ def.num_classes+"}");
                    }

                    else if (def.type ==  "regression ")
                    {
                        // add an fc layer here, there is no reason the user should
                        // have to worry about this and we almost always want to
                        new_defs.Add("{ type:  'fc', num_neurons:"+ def.num_neurons+"}");
                    }
                    else if (def.type == "fc" || def.type == "conv")
                    {
                        def.bias_pref = 0.0;
                        if (def.activation != null && def.activation == "relu")
                        {
                            def.bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
                            // otherwise it "s technically possible that a relu unit will never turn on (by chance)
                            // and will never get any gradient and never contribute any computation. Dead relu.
                        }
                    }
                        new_defs.Add(defs[i]);
                    

                    if ( def.activation != null)
                    {
                        if (def.activation ==  "relu") { new_defs.Add("{ type: 'relu'}"); }
                        else if (def.activation ==  "sigmoid") { new_defs.Add("{ type: 'sigmoid'}"); }
                        else if (def.activation ==  "tanh") { new_defs.Add("{ type: 'tanh'}"); }
                        else if (def.activation ==  "maxout")
                        {
                            // create maxout activation, and pass along group size, if provided
                            var gs = def.group_size !=  0 ? def.group_size : 2;
                            new_defs.Add("{ type: 'maxout', group_size: "+gs+"}");
                        }
                        else
                        {
                            Console.WriteLine("ERROR unsupported activation " + def.activation);
                        }
                    }
                    if ( def.drop_prob != null && def.type !="dropout")
                    {
                        new_defs.Add("{ type: 'dropout', drop_prob: "+def.drop_prob+"}");
                    }

                }
        
             defs = new_defs;

              // create the layers
              for(var i = 0;i<defs.Count;i++) 
              {
                Layer_def def = JsonConvert.DeserializeObject<Layer_def>(defs[i]);
                if (i > 0)
                {
                    var prev = this.layers[i - 1];
                    def.in_sx = prev.out_sx;
                    def.in_sy = prev.out_sy;
                    def.in_depth = prev.out_depth;
                }

                switch (def.type)
                {
                    case "fc": this.layers.Add(new FullyConnLayer(def)); break;
                    case "lrn": this.layers.Add(new LocalResponseNormalizationLayer(def)); break;
                    case "dropout": this.layers.Add(new DropoutLayer(def)); break;
                    case "input": this.layers.Add(new InputLayer(def)); break;
                    case "softmax": this.layers.Add(new SoftmaxLayer(def)); break;
                    case "regression": this.layers.Add(new RegressionLayer(def)); break;
                    case "conv": this.layers.Add(new ConvLayer(def)); break;
                    case "pool": this.layers.Add(new PoolLayer(def)); break;
                    case "relu": this.layers.Add(new ReluLayer(def)); break;
                    case "sigmoid": this.layers.Add(new SigmoidLayer(def)); break;
                    case "tanh": this.layers.Add(new TanhLayer(def)); break;
                    case "maxout": this.layers.Add(new MaxoutLayer(def)); break;
                    case "svm": this.layers.Add(new SVMLayer(def)); break;
                    default: Console.WriteLine("ERROR: UNRECOGNIZED LAYER TYPE: " + def.type); break;
                }
            }
         }

        // forward prop the network. 
        // The trainer class passes is_training = true, but when this function is
        // called from outside (not from the trainer), it defaults to prediction mode
        public Vol forward(Vol V, bool is_training)
        {
            var act = this.layers[0].forward(V, is_training);
            for (var i = 1; i < this.layers.Count; i++)
            {
                act = this.layers[i].forward(act, is_training);
            }
            return act;
        }

        public double backward (int y)
        {
            var N = this.layers.Count;
            var loss = this.layers[N - 1].backward(y); // last layer assumed to be loss layer
            for (var i = N - 2; i >= 0; i--)
            { // first layer assumed input
                this.layers[i].backward();
            }
            return loss;
        }
        
        
        public double getCostLoss(Vol V, int y)
        {
            this.forward(V, false);
            var N = this.layers.Count;
            var loss = this.layers[N - 1].backward(y);
            return loss;
        }



        public List<params_and_grads> getParamsAndGrads()
        {
            // accumulate parameters and gradients for the entire network
            List<params_and_grads> response = new List<params_and_grads>();
            for (var i = 0; i < this.layers.Count; i++)
            {
                var layer_reponse = this.layers[i].getParamsAndGrads();
                for (var j = 0; j < layer_reponse.Count; j++)
                {
                    response.Add(layer_reponse[j]);
                }
            }
            return response;
        }
        public int getPrediction()
        {
            // this is a convenience function for returning the argmax
            // prediction, assuming the last layer of the net is a softmax
            var S = this.layers[this.layers.Count - 1];
           //  assert(S.layer_type == "softmax", 'getPrediction function assumes softmax as last layer of the net!');

            var p = S.out_act.w;
            var maxv = p[0];
            var maxi = 0;
            for (var i = 1; i < p.Length; i++)
            {
                if (p[i] > maxv) { maxv = p[i]; maxi = i; }
            }
            return maxi; // return index of the class with highest class probability
        }
        
    }
}
