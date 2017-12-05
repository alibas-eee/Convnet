using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace convnet
{
    class Convnet_util
    {
        bool return_v = false;
        double v_val = 0.0;
        Random rndm = new Random();
        public double gaussRandom()
        { 
            if(return_v) { 
              return_v = false;
              return v_val; 
            }
            
            var u = 2 * rndm.NextDouble() - 1;
            var v = 2 * rndm.NextDouble() - 1;
            var r = u * u + v * v;
            if(r == 0 || r > 1) return gaussRandom();
            var c = Math.Sqrt(-2 * Math.Log(r) / r);
            v_val = v* c; // cache this
            return_v = true;
            return u* c;
        }
        public double randf(double a, double b) { return rndm.NextDouble() * (b-a)+a; }
        public double randi(double a, double b) { return Math.Floor(rndm.NextDouble() * (b-a)+a); }
        public double randn(double mu, double std){ return mu+gaussRandom()* std; }

        public static Vol img_to_vol(Bitmap img, bool convert_grayscale)
        {
            ImageConverter converter = new ImageConverter();

            var p = (byte[])converter.ConvertTo(img, typeof(byte[]));
            var W = img.Width;
            var H = img.Height;
            Stack<double> pv = new Stack<double>();
            for (var i = 0; i < p.Length; i++)
            {
                pv.Push(p[i] / 255.0 - 0.5); // normalize image pixels to [-0.5, 0.5]
            }
            var x = new Vol(W, H, 4, 0.0); //input volume (image)
            x.w = pv.ToArray();

            if (convert_grayscale)
            {
                // flatten into depth=1 array
                var x1 = new Vol(W, H, 1, 0.0);
                for (var i = 0; i < W; i++)
                {
                    for (var j = 0; j < H; j++)
                    {
                        x1.set(i, j, 0, x.get(i, j, 0));
                    }
                }
                x = x1;
            }


            return x;
        }
        // Array utilities
        public static double[] zeros(int n)
        {
            //if(typeof(n)=='undefined' || isNaN(n)) { return []; }
            // lacking browser support
            var arr = new double[n];
              for(var i = 0; i<n;i++) { arr[i]= 0; }
              return arr;
        }
        public static int[] zeros_int(int n)
        {
            //if(typeof(n)=='undefined' || isNaN(n)) { return []; }
            // lacking browser support
            var arr = new int[n];
            for (var i = 0; i < n; i++) { arr[i] = 0; }
            return arr;
        }
        public static bool[] zeros_bool(int n)
        {
            //if(typeof(n)=='undefined' || isNaN(n)) { return []; }
            // lacking browser support
            var arr = new bool[n];
            for (var i = 0; i < n; i++) { arr[i] = false; }
            return arr;
        }

        bool arrContains (double [] arr,double elt)
        {
            for(int i = 0, n = arr.Length; i<n;i++) {
              if(arr[i]==elt) return true;
            }
            return false;
        }

        double [] arrUnique(double []arr)
        {
            Stack<double> b = new Stack<double>();
            for(int i = 0, n = arr.Length; i<n;i++) {
              if(!arrContains(b.ToArray(), arr[i])) {
                b.Push(arr[i]);
              }
            }
            return b.ToArray();
        }

  // return max and min of a given non-empty array.
/*  var maxmin = function(w) {
        if(w.length === 0) { return {}; } // ... ;s
        var maxv = w[0];
    var minv = w[0];
    var maxi = 0;
    var mini = 0;
    var n = w.length;
        for(var i = 1; i<n;i++) {
          if(w[i] > maxv) { maxv = w[i]; maxi = i; } 
          if(w[i] < minv) { minv = w[i]; mini = i; } 
        }
        return {maxi: maxi, maxv: maxv, mini: mini, minv: minv, dv:maxv-minv};
  }*/

  // create random permutation of numbers, in range [0...n-1]
    public double[] randperm (int n) {
        int i = n,
            j = 0;
        double  temp;
        double [] array=new double[n];
        for(var q = 0; q<n;q++)array[q]=q;
            while (i > 0 ) {
            i--;
            j = (int) Math.Floor(rndm.NextDouble() * (i+1));
            temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
        return array;
      }
        /*
  // sample from list lst according to probabilities in list probs
  // the two lists are of same size, and probs adds up to 1
  public double weightedSample(List<double> lst,double [] probs) 
  {
    double p = rndm.NextDouble();
    double cumprob = 0.0;
        for(int k = 0, n = lst.Count; k<n;k++) 
            {
          cumprob += probs[k];
          if(p<cumprob) { return lst[k]; }
        }
            return 0;
  }

  // syntactic sugar function for getting default parameter values
  var getopt = function(opt, field_name, default_value) {
    if(typeof field_name === 'string') {
      // case of single string
      return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
    } else {
      // assume we are given a list of string instead
      var ret = default_value;
      for(var i = 0; i<field_name.length;i++) {
        var f = field_name[i];
        if (typeof opt[f] !== 'undefined') {
          ret = opt[f]; // overwrite return value
        }
      }
      return ret;
    }
  }

  function assert(condition, message)
{
    if (!condition)
    {
        message = message || "Assertion failed";
        if (typeof Error !== "undefined")
        {
            throw new Error(message);
        }
        throw message; // Fallback
    }
}

    */

    }
}
