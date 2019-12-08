#include "CatDogCNN.h"
using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

Status CatDogCNN::CreateGraphForImage(bool unstack)
{
	file_name_var = Placeholder(i_root.WithOpName("input"), DT_STRING);
	auto file_reader = ReadFile(i_root.WithOpName("file_readr"), file_name_var);

	auto image_reader = DecodeJpeg(i_root.WithOpName("jpeg_reader"), 
																file_reader, DecodeJpeg::Channels(image_channels));
	auto float_caster = Cast(i_root.WithOpName("float_caster"), image_reader, DT_FLOAT);
	auto dims_exapnder = ExapndDims(i_root.WithOpName("dim"), float_caster, 0);

	auto resized = ResizeBilinear(i_root.WithOpName("size"), 
																dims_exapnder, Const(i_root, {image_side, image_side}));
	auto div = Div(i_root.WithOpName("normalized"), resized, {255.f});
	if(unstack)
	{
		auto output_list = Unstack(i_root.WithOpName("fold"), div, 1);
		image_tensor_var = output_list.output[0];
	}
	else
	{
		image_tensor_var = div;
	}
	return i_root.status();
}

Status CatDogCNN::ReadTensorFromImageFile(string& file_name, Tensor& outTensor)
{
	if(!i_root.ok())
		return i_root.status();
	if (!absl::EndsWith(file_name, ".jpg") && !absl::EndsWith(file_name, ".jpeg"))
	{
		return errors::InvalidArgument("Image must be jpeg encoded");
	}
	vector<Tensor> out_tensors;
	ClientSession session(i_root);
	TF_CHECK_OK(session.Run({{file_name_var, file_name}}, {image_tensor_var}, &out_tensors));
	outTensor = out_tensors[0]; // shallow copy
	return Status::OK();
}





Status CatDogCNN::ReadFileTensors(string& base_folder_name, vector<pair<string, float>> v_folder_label, vector<pair<Tensor, float>>& file_tensors)
{
	//validate the folder
	Env* penv = Env::Default();
	TF_RETURN_IF_ERROR(penv->IsDirectory(base_folder_name));
	//get the files
	bool b_shuffle = false;
	for(auto &&p: v_folder_label)
	{
		string folder_name = io::JoinPath(base_folder_name, p.first);
		TF_RETURN_IF_ERROR(penv->IsDirectory(folder_name));
		vector<string> file_names;
		TF_RETURN_IF_ERROR(penv->GetChildren(folder_name, &file_names));
		for(string file: file_names)
		{
			string full_path = io::JoinPath(folder_name, file);
			Tensor i_tensor;
			TF_RETURN_IF_ERROR(ReadTensorFromImageFile(full_path, i_tensor));
			size_t s = file_tensors.size();
			if(b_shuffle)
			{
				//suffle the images
				int i = rand() % s;
				file_tensors.emplace(file_tensors.begin()+i, make_pair(i_tensor, p.second));
			}
			else
				file_tensors.push_back(make_pair(i_tensor, p.second));
		}
		b_shuffle = true;
	}
	return Status::OK();
}
