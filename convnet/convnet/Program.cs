using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace convnet
{
    class Program
    {
        static void Main(string[] args)
        {

            List<String> layer_defs = new List<String>();

            /*
            layer_defs.Add("{ type: 'input', out_sx: 24, out_sy: 24, out_depth: 1}");
            layer_defs.Add("{ type: 'conv', sx: 5, filters: 8, stride: 1, pad: 2, activation: 'relu'}");
            layer_defs.Add("{ type: 'pool', sx: 2, stride: 2}");
            layer_defs.Add("{ type: 'conv', sx: 5, filters: 16, stride: 1, pad: 2, activation: 'relu'}");
            layer_defs.Add("{ type: 'pool', sx: 3, stride: 3}");
            layer_defs.Add("{ type: 'softmax', num_classes: 10}");
            */

            layer_defs.Add("{type:'input', out_sx:1, out_sy:1, out_depth:2}");
            layer_defs.Add("{type:'fc', num_neurons:2, activation: 'sigmoid',bias_pref:1}");
            layer_defs.Add("{type:'softmax', num_classes:2}");
            


            Net net = new Net();
            net.makeLayers(layer_defs);
            String json_trainer = "{learning_rate:0.1, momentum:0.9, batch_size:10, l2_decay:0.1,method:'adadelta'}";
            Trainer trainer = new Trainer(net, json_trainer);

            var x = new Vol(1, 1, 2);

            double[][] data = new double[][]
            {
                new double[] {0,0},
                new double[] {0,1},
                new double[] {1,0},
                new double[] {1,1}
            };
            int[] label = new int[]{
                0,1,1,0
            };
            for (int i = 0; i < 4000; i++)
            {
                int j = i % 4;
                x.w = data[j];
                String tr = trainer.train(x, label[j]);
                //Console.WriteLine(tr);
                Console.WriteLine(net.layers[3].out_act.dw[0].ToString());
            }
            
           
            while (true) ;
        }
    }
}
