#include "postprocess.h"
#include <algorithm>
#include <math.h>


static inline float fast_exp(float x)
{
    // return exp(x);
    union
    {
        uint32_t i;
        float f;
    } v;
    v.i = (12102203.1616540672 * x + 1064807160.56887296);
    return v.f;
}


static float DeQnt2F32(int8_t qnt, int zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}


GetResultRectyolo26::GetResultRectyolo26()
{
}

GetResultRectyolo26::~GetResultRectyolo26()
{
}

float GetResultRectyolo26::sigmoid(float x)
{
    return 1 / (1 + fast_exp(-x));
}

int GetResultRectyolo26::GenerateMeshgrid()
{
    int ret = 0;
    if (headNum == 0)
    {
        printf("=== yolo26 Meshgrid  Generate failed! \n");
    }

    for (int index = 0; index < headNum; index++)
    {
        for (int i = 0; i < mapSize[index][0]; i++)
        {
            for (int j = 0; j < mapSize[index][1]; j++)
            {
                meshgrid.push_back(float(j + 0.5));
                meshgrid.push_back(float(i + 0.5));
            }
        }
    }

    printf("=== yolo26 Meshgrid  Generate success! \n");

    return ret;
}

int GetResultRectyolo26::GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects)
{
    int ret = 0;
    if (meshgrid.empty())
    {
        ret = GenerateMeshgrid();
    }

    int gridIndex = -2;
    float xmin = 0, ymin = 0, xmax = 0, ymax = 0;
    float cx = 0, cy = 0, cw = 0, ch = 0;

    float cls_val = 0;
    float cls_max = 0;
    int cls_index = 0;

    int quant_zp_cls = 0, quant_zp_reg = 0;
    float quant_scale_cls = 0, quant_scale_reg = 0;

    for (int index = 0; index < headNum; index++)
    {
        int8_t *reg = (int8_t *)pBlob[index * 2 + 0];
        int8_t *cls = (int8_t *)pBlob[index * 2 + 1];

        quant_zp_reg = qnt_zp[index * 2 + 0];
        quant_zp_cls = qnt_zp[index * 2 + 1];

        quant_scale_reg = qnt_scale[index * 2 + 0];
        quant_scale_cls = qnt_scale[index * 2 + 1];

        for (int h = 0; h < mapSize[index][0]; h++)
        {
            for (int w = 0; w < mapSize[index][1]; w++)
            {
                gridIndex += 2;

                if (1 == class_num)
                {
                    cls_max = sigmoid(DeQnt2F32(cls[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_cls, quant_scale_cls));
                    cls_index = 0;
                }
				else
				{
					for (int cl = 0; cl < class_num; cl++)
					{
						cls_val = cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w];

						if (0 == cl)
						{
							cls_max = cls_val;
							cls_index = cl;
						}
						else
						{
							if (cls_val > cls_max)
							{
								cls_max = cls_val;
								cls_index = cl;
							}
						}
					}
					cls_max = sigmoid(DeQnt2F32(cls_max, quant_zp_cls, quant_scale_cls));
				}

                if (cls_max > objectThresh)
                {
                    cx = DeQnt2F32(reg[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_reg, quant_scale_reg);
                    cy = DeQnt2F32(reg[1 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_reg, quant_scale_reg);
                    cw = DeQnt2F32(reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_reg, quant_scale_reg);
                    ch = DeQnt2F32(reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w], quant_zp_reg, quant_scale_reg);
                    
                    xmin = (meshgrid[gridIndex + 0] - cx) * strides[index];
                    ymin = (meshgrid[gridIndex + 1] - cy) * strides[index];
                    xmax = (meshgrid[gridIndex + 0] + cw) * strides[index];
                    ymax = (meshgrid[gridIndex + 1] + ch) * strides[index];

                    xmin = xmin > 0 ? xmin : 0;
                    ymin = ymin > 0 ? ymin : 0;
                    xmax = xmax < input_w ? xmax : input_w;
                    ymax = ymax < input_h ? ymax : input_h;

                    if (xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h)
                    {
                        DetectRect temp;
                        temp.xmin = xmin / input_w;
                        temp.ymin = ymin / input_h;
                        temp.xmax = xmax / input_w;
                        temp.ymax = ymax / input_h;
                        temp.classId = cls_index;
                        temp.score = cls_max;

                        DetectiontRects.push_back(temp.classId);
                        DetectiontRects.push_back(temp.score);
                        DetectiontRects.push_back(temp.xmin);
                        DetectiontRects.push_back(temp.ymin);
                        DetectiontRects.push_back(temp.xmax);
                        DetectiontRects.push_back(temp.ymax);
                    }
                }
            }
        }
    }

    return ret;
}
