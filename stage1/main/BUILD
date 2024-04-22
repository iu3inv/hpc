load("@rules_cc//cc:defs.bzl", "cc_binary")
# load("@rules_cuda//cuda:defs.bzl", "cuda_library")


cc_binary(
    name = "hello-world",
    srcs = ["hello-world_main.cc"],
    deps = ["@com_github_google_glog//:glog"
             ],
)

#cuda_library(
#    name = "hello-world-cuda",
#    srcs = ["hello.cu"],
#)

cc_binary(
    name = "t",
    srcs = ["main.cu"],
    deps = ["@cuda//:cuda_runtime"],
)
