using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace convnet
{
    class params_and_grads
    {
        public double l1_decay_mul;
        public double l2_decay_mul;
        public double[] param;
        public double[] grads;
        public params_and_grads()
        {
            l1_decay_mul = 0;
            l2_decay_mul = 0;
            param = new double[0];
            grads = new double[0];
        }
    }

    abstract class layer
    {
        public int out_depth{set; get;}
        public int out_sx { set; get; }
        public int out_sy { set; get; }

        public int num_inputs { set; get; }
        public int in_depth { set; get; }
        public int in_sx { set; get; }
        public int in_sy { set; get; }

        public double bias { set; get; }
        public double l1_decay_mul { set; get; }
        public double l2_decay_mul { set; get; }

        public String layer_type { set; get; }

        public Vol out_act { set; get; }
        public Vol in_act  { set; get; }
        public Vol biases { set; get; }
        public List<Vol> filters { set; get; }

        public List<params_and_grads> response;

        public layer()
        {
            response = new List<params_and_grads>();
        }

        public abstract Vol forward(Vol V, bool is_training);

        public abstract void backward();
        public abstract double backward(int u);        
        public abstract List<params_and_grads> getParamsAndGrads();


    }
}
