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
CPPFLAGS += `pkg-config --cflags protobuf grpc lunasdk`
CXXFLAGS += -std=c++11

LDFLAGS += -L/usr/local/lib -L/usr/lib `pkg-config --libs protobuf grpc++ grpc lunasdk`\
           -lgrpc++_reflection\
           -ldl\
           -rdynamic /usr/lib/libFaceEngineSDK.so\
           -Wl,-rpath,/home/luna-sdk_linux_rel_v.3.6.6/examples/lib/gcc4/x64:

LDFLAGS_ += -L/usr/local/lib -L/usr/lib `pkg-config --libs protobuf grpc++ grpc lunasdk`\
           -Wl,--no-as-needed -lgrpc++_reflection -Wl,--as-needed\
           -ldl


all:   greeter_server	

greeter_server: test_api.pb.o test_api.grpc.pb.o greeter_server.o
	$(CXX) $^ $(LDFLAGS) -o $@

server:  test_api.pb.o test_api.grpc.pb.o greeter_server.o
	$(CXX) $^ $(LDFLAGS_) -o $@

clean:
	rm -f *.o greeter_server
	

.PHONY: all clean server


