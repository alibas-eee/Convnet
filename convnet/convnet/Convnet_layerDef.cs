using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace convnet
{
    class Layer_def
    {
       public String type      { get; set; }
       public String drop_prob { get; set; }
       public int sx { get; set; }
       public int sy { get; set; }
       public int stride { get; set; }
       public int pad { get; set; }
       public int out_depth { get; set; }
       public int num_classes { get; set; }
       public int num_neurons { get; set; }
       public int filters { get; set; }
       public int in_sx { get; set; }
       public int in_sy { get; set; }
       public int out_sx { get; set; }
       public int out_sy { get; set; }
       public int in_depth { get; set; }                    
       public double l1_decay_mul { get; set; }
       public double l2_decay_mul { get; set; }
       public double bias_pref { get; set; }
       public string activation {get; set; }
       public int group_size { get; set; }

        public int k { get; set; }
        public int n { get; set; }
        public double alpha { get; set; }
        public double beta { get; set; }

        
       public Layer_def()
       {
           l1_decay_mul = 0;
           l2_decay_mul = 0;
           bias_pref    = 1.0;
       }
    }
}
