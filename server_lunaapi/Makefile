#
# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

CXX = g++
CPPFLAGS += `pkg-config --cflags protobuf grpc`
CXXFLAGS += -std=c++11
ifeq ($(SYSTEM),Darwin)
LDFLAGS += -L/usr/local/lib `pkg-config --libs protobuf grpc++ grpc`\
           -lgrpc++_reflection\
           -ldl
else
LDFLAGS += -L/usr/local/lib `pkg-config --libs protobuf grpc++ grpc`\
           -Wl,--no-as-needed -lgrpc++_reflection -Wl,--as-needed\
           -ldl
endif

PROTOC =../../grpc-1.16.0/grpc/bins/opt/protoc
GRPC_CPP_PLUGIN = grp_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= 'which $(GRPC_CPP_PLUSIN)'

PROTOS_PATH = $(CURDIR)


all:   greeter_server	

greeter_server: test_api.pb.o test_api.grpc.pb.o greeter_server.o
	$(CXX) $^ $(LDFLAGS) -o $@

.PRECIOUS: %.grpc.pb.cc
%.grpc.pb.cc: %.proto
	$(PROTOC) -I $(PROTOS_PATH) --grpc_out=. --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH) $<

.PRECIOUS: %.grpc.pb.cc
%.grpc.pb.cc:%.proto
	$(PROTOC) -I $(PROTOS_PATH) --CPP_out=. $<
	
clean:
	rm -f *.o *.pb.cc *.pb.h greeter_server
	
setpkg:
    shell export PKG_CONFIG_PATH=/usr/lib/pkgconfig:/usr/lib/pkgconfig

.PHONY: all clean setpkg


