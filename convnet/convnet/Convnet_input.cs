using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace convnet
{
    class InputLayer : layer 
    {       
        public InputLayer(Layer_def opt)
        {            
            // required: depth
            out_depth = opt.out_depth;

            // optional: default these dimensions to 1
            out_sx = opt.out_sx != 0 ? opt.out_sx : 1;
            out_sy = opt.out_sy != 0 ? opt.out_sy : 1;

            // computed
            layer_type = "input";
        }
        public override Vol forward(Vol V,bool is_training)
        {
            this.in_act = V;
            this.out_act = V;
            return this.out_act; // simply identity function for now
        }
        public override void backward() { }
        public override double backward(int y) { return 0; }
        public override List<params_and_grads> getParamsAndGrads()
        {
            return response;
        }
    }
}
