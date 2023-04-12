using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace onnx_supervised
{
    public class BurialData
    {
        public float depth { get; set; }
        public float length { get; set; }
        public float area_NE { get; set; }
        public float area_NW { get; set; }
        public float area_SE { get; set; }
        public float area_SW { get; set; }
        public float wrapping_B { get; set; }
        public float wrapping_H { get; set; }
        public float wrapping_W { get; set; }
        public float ageatdeath_A { get; set; }
        public float ageatdeath_C { get; set; }
        public float ageatdeath_I { get; set; }
        public float ageatdeath_IN { get; set; }
        public float ageatdeath_N { get; set; }

        public Tensor<float> AsTensor()
        {
            float[] data = new float[]
            {
                depth, length, area_NE, area_NW, area_SE, area_SW, wrapping_B, wrapping_H, wrapping_W,
                ageatdeath_A, ageatdeath_C, ageatdeath_I, ageatdeath_IN, ageatdeath_N
            };
            int[] dimensions = new int[] { 1, 14 };
            return new DenseTensor<float>(data, dimensions);
        }
    }
}
