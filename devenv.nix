{ pkgs, lib, ... }:
let
  buildInputs = with pkgs; [
    # cudaPackages.cuda_cudart
    # cudaPackages.cudatoolkit
    # cudaPackages.cudnn
    stdenv.cc.cc
    libuv
    zlib
  ];
in
{
  # packages = with pkgs; [
  #   cudaPackages.cuda_nvcc
  # ];

  env = {
    LD_LIBRARY_PATH = "${lib.makeLibraryPath buildInputs}:/run/opengl-driver/lib:/run/opengl-driver-32/lib";

    # XLA_FLAGS =
    #   "--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit} "
    #   + "--xla_gpu_triton_gemm_any=True "
    #   + "--xla_gpu_enable_custom_fusions=true "
    #   + "--xla_gpu_enable_cudnn_fmha=false "
    #   + "--xla_gpu_enable_latency_hiding_scheduler=true ";
    # XLA_PYTHON_CLIENT_MEM_FRACTION = 0.9;
    # CUDA_PATH = pkgs.cudaPackages.cudatoolkit;
  };

  languages.python = {
    enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  enterShell = ''
    . .devenv/state/venv/bin/activate
  '';

}
