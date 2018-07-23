#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <tuple>
#include <fstream>
#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/math_functions.hpp"
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using caffe::NetParameter;
using caffe::LayerParameter;
using caffe::caffe_scal;
using caffe::caffe_powx;
using caffe::caffe_mul;
using caffe::caffe_add;
using caffe::caffe_div;
using caffe::caffe_copy;
using caffe::caffe_add_scalar;

using std::tuple;
using std::make_tuple;
using std::ostringstream;


std::vector<std::string> splitString(const std::string& src,const std::string &symbols,bool repeat)
 {
     std::vector<std::string> result;
     int startIdx=0;
     for (int i=0;i<src.length();i++)
     {
         bool isMatch=false;
         for (int j=0;j<symbols.length();j++)
         {
             if (src[i]==symbols[j])
             {
                 isMatch=true;
                 break;
             }
             if (!repeat)
             {
                 break;
             }
         }
         if (isMatch)
         {
             std::string sub=src.substr(startIdx,i-startIdx);
             startIdx=i+1;
             if (sub.length()>0)
             {             
                 result.push_back(sub);
             }             
         }
         if (i+1==src.length())
         {
             std::string sub=src.substr(startIdx,src.length()-startIdx);
             startIdx=i+1;
             if (sub.length()>0)
             {             
                 result.push_back(sub);
             }
         }
     }     
     return result; 
 }

 
bool isIdxBatchScale(const std::vector<std::string>&lines,int startLineIdx,int endLineIdx)
{
    for (int j=startLineIdx;j<endLineIdx;j++)
    {
        std::vector<std::string> segs=splitString(lines[j]," :\15\32",true);
        if(std::find(segs.begin(),segs.end(),"\"BatchNorm\"")!=segs.end()
            ||std::find(segs.begin(),segs.end(),"\"Scale\"")!=segs.end()
            ) 
        {
            return true;    
        }
    }
    return false;
}


int merge_bn()
{
    std::string FLAGS_model="/media/hdd/lbl_trainData/dataBase/ssdpelee/work/jobs/vehiclePeroson/deploy.prototxt";
    std::string FLAGS_weights="/media/hdd/lbl_trainData/dataBase/ssdpelee/work/models/vehiclePeroson/ssd304p_vehiclePeroson_iter_15000.caffemodel";
    //FLAGS_model="vgg_test.prototxt";
    //FLAGS_weights="vgg_test.caffemodel";
      LOG(INFO) << "Use CPU.";
      Caffe::set_mode(Caffe::CPU);
    vector<std::string> stages;
      Net<float> caffe_net(FLAGS_model, caffe::TEST, 0, &stages);
      caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
    NetParameter param;
    ReadNetParamsFromTextFileOrDie(FLAGS_model, &param);
    LOG(INFO)<<"net name : "<<param.name();
    LOG(INFO)<<"net layers size : "<<param.layer_size();
#ifdef DEBUG
    const vector<std::string>&layersFromWeight = caffe_net.layer_names();
#endif
    vector<tuple<const shared_ptr<Layer<float>> ,const shared_ptr<Layer<float>> ,const shared_ptr<Layer<float>> >>convBNs;
    for(int i=0;i<param.layer_size()-2;i++)
    {
        const LayerParameter& layer_param = param.layer(i);
#ifdef DEBUG
        LOG(INFO)<<i<<" thisLayers name : "<<layer_param.name()<<"\t"<<layersFromWeight[i]<<"\t"<<layer_param.type();
        //LOG(INFO)<<i<<" thisLayers name : "<<caffe_net.layers()[i]->type();
#endif 
        if(std::string(layer_param.type()).compare("Convolution")==0
        && std::string(param.layer(i+1).type()).compare("BatchNorm")==0
        && std::string(param.layer(i+2).type()).compare("Scale")==0)
            //convBNs.push_back(make_tuple( caffe_net.layers()[i],caffe_net.layers()[i+1],caffe_net.layers()[i+2]            ));
            convBNs.push_back(make_tuple( caffe_net.layer_by_name(param.layer(i).name()),
                                          caffe_net.layer_by_name(param.layer(i+1).name()),
                                          caffe_net.layer_by_name(param.layer(i+2).name())
                                          ));
    }
    for(auto&convBN : convBNs)
    {
        auto convPtr=std::get<0>(convBN);
        auto bnPtr=std::get<1>(convBN);
        auto scalePtr=std::get<2>(convBN);
        LOG(INFO)<<"conv:"<<convPtr->blobs().size()<<" bn:"<<bnPtr->blobs().size()<<" scale:"<<scalePtr->blobs().size();
        auto&convWeightBlobPtr=convPtr->blobs()[0];
        auto&bnMeanBlobPtr=bnPtr->blobs()[0];
        auto&bnVarBlobPtr=bnPtr->blobs()[1];
        auto&bnScaleBlobPtr=bnPtr->blobs()[2];
        auto&bnABlobPtr=scalePtr->blobs()[0];
        auto&bnBBlobPtr=scalePtr->blobs()[1];
        CHECK(convPtr->blobs().size() == 1);
        CHECK(convWeightBlobPtr->num() == bnMeanBlobPtr->num());
        CHECK(convWeightBlobPtr->num() == bnVarBlobPtr->num());
        CHECK(1 == bnScaleBlobPtr->num());
        CHECK(convWeightBlobPtr->num() == bnABlobPtr->num());
        CHECK(convWeightBlobPtr->num() == bnBBlobPtr->num());
        float scale_ = bnScaleBlobPtr->cpu_data()[0];
#ifdef DEBUG
        LOG(INFO)<<"\t\tmean_ = "<<bnMeanBlobPtr->cpu_data()[0] ;
        LOG(INFO)<<"\t\tvar_ = "<<bnVarBlobPtr->cpu_data()[0] ;
        LOG(INFO)<<"\t\tSCALE_ = "<<scale_ ;
#endif
        if(scale_<0.01)
        scale_=1;
        else
        scale_=1/scale_;
        caffe_scal(bnMeanBlobPtr->count(), scale_, bnMeanBlobPtr->mutable_cpu_data());
        caffe_scal(bnMeanBlobPtr->count(), float(-1.0), bnMeanBlobPtr->mutable_cpu_data());
        caffe_scal(bnMeanBlobPtr->count(), scale_, bnVarBlobPtr->mutable_cpu_data());
        caffe_add_scalar(bnMeanBlobPtr->count(), float(0.0010000000), bnVarBlobPtr->mutable_cpu_data());
        caffe_powx(bnMeanBlobPtr->count(), bnVarBlobPtr->cpu_data(), float(0.5),bnVarBlobPtr->mutable_cpu_data());
#ifdef DEBUG
        LOG(INFO)<<"\t\tmean_ = "<<bnMeanBlobPtr->cpu_data()[0] ;
        LOG(INFO)<<"\t\tvar_ = "<<bnVarBlobPtr->cpu_data()[0] ;
#endif
        int inner_count=convWeightBlobPtr->count()/convWeightBlobPtr->num();
        LOG(INFO)<<"output :"<<bnMeanBlobPtr->count();
        for(int row=0;row<bnMeanBlobPtr->count();row++)
        {
            float*convWeight=convWeightBlobPtr->mutable_cpu_data()+convWeightBlobPtr->offset(row,0,0,0);
            caffe_scal(inner_count, bnABlobPtr->cpu_data()[row], convWeight);
            caffe_scal(inner_count, float(1.0)/(bnVarBlobPtr->cpu_data()[row]), convWeight);
        }
        convPtr->blobs().push_back(shared_ptr<Blob<float>>( new Blob<float>(bnABlobPtr->shape())     ));
        auto&convBiasBlobPtr=convPtr->blobs()[1];
        caffe_copy(bnABlobPtr->num(), bnABlobPtr->cpu_data(), convBiasBlobPtr->mutable_cpu_data());
        caffe_mul(bnABlobPtr->num(), convBiasBlobPtr->cpu_data(), bnMeanBlobPtr->cpu_data(), convBiasBlobPtr->mutable_cpu_data());
        caffe_div(bnABlobPtr->num(), convBiasBlobPtr->cpu_data(), bnVarBlobPtr->cpu_data(), convBiasBlobPtr->mutable_cpu_data());
        caffe_add(bnABlobPtr->num(), bnBBlobPtr->cpu_data(), convBiasBlobPtr->cpu_data(), convBiasBlobPtr->mutable_cpu_data());
        LOG(INFO)<<"conv:"<<convPtr->blobs().size()<<" bn:"<<bnPtr->blobs().size()<<" scale:"<<scalePtr->blobs().size();
        LOG(INFO)<<"conv:"<<convPtr->blobs()[0]->shape_string()<<"   "<<convPtr->blobs()[1]->shape_string();
    }
    for(int i=0;i<param.layer_size()-2;i++)    LOG(INFO)<<i<<" thisLayers type : "<<caffe_net.layers()[i]->type()<<" size="<<caffe_net.layers()[i]->blobs().size();
    NetParameter net_param;
    caffe_net.ToProto(&net_param, false);
    WriteProtoToBinaryFile(net_param, "pelee_nobn.caffemodel");
    std::fstream fin(FLAGS_model,std::ios::in);
    std::fstream fout("pelee_nobn.prototxt",std::ios::out);    
    std::vector<int> layerLineIdx;
    std::vector<std::string> lines;
    std::string aline;
    int lineIdx=0;
    std::string thisLayerType="";
    while(std::getline(fin,aline))
    {
        std::vector<std::string> segs=splitString(aline," :\15\32",true);
        if(std::find(segs.begin(),segs.end(),"layer")!=segs.end()) layerLineIdx.push_back(lineIdx);
        lines.push_back(aline);
        lineIdx++;
    }
    layerLineIdx.push_back(lineIdx);
    for (int i=0;i<layerLineIdx[0];i++)
    {
        fout<<lines[i]<<std::endl;
    }
    for (int i=0;i<layerLineIdx.size()-1;i++)
    {
        int scopeStrat=layerLineIdx[i];
        int scopeEnd=layerLineIdx[i+1];
        bool isNextBNorScale=false;
        for (int j=scopeStrat;j<scopeEnd;j++)
        {
            
            //std::cout<<"-----------------"<<lines[j]<<std::endl;
            std::vector<std::string> segs=splitString(lines[j]," :\15\32",true);
            if(std::find(segs.begin(),segs.end(),"\"Convolution\"")!=segs.end()) 
            {                
                if (i<layerLineIdx.size()-3)
                {
                    int potentialBNstrat=layerLineIdx[i+1];
                    int potentialBNend=layerLineIdx[i+2];
                    int potentialScaleStrat=layerLineIdx[i+2];
                    int potentialScaleEnd=layerLineIdx[i+3];
                    isNextBNorScale=isIdxBatchScale(lines,potentialBNstrat,potentialBNend)&&isIdxBatchScale(lines,potentialScaleStrat,potentialScaleEnd);
                }
            }
        }
            if (isNextBNorScale)
            {
                i+=2;
                for (int q=scopeStrat;q<scopeEnd;q++)
                {
                    std::vector<std::string> segs=splitString(lines[q]," :\15\32",true);
                    if(std::find(segs.begin(),segs.end(),"bias_term")!=segs.end()
                        &&std::find(segs.begin(),segs.end(),"false")!=segs.end()) 
                    continue;
                    fout<<lines[q]<<std::endl;
                    //std::cout<<lines[q]<<std::endl;
                }
            }
            else
            {
                for (int q=scopeStrat;q<scopeEnd;q++)
                {
                    fout<<lines[q]<<std::endl;
                    //std::cout<<lines[q]<<std::endl;
                }
            }        
    }
    fin.close();
    fout.close();

    return 0;
}

int main()
{
    merge_bn();
    return 0;
}
