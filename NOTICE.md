## NOTICE

This project is a fork of [https://github.com/codelion/optillm/tree/main](https://github.com/codelion/optillm/tree/main)

The [original project](https://github.com/codelion/optillm/tree/main) is licensed under the Apache 2 License as detailed in [the original LICENSE](https://github.com/codelion/optillm/blob/main/LICENSE). The fork was created from a December 2024 fork of the original project.


This project, i.e., [CePO](https://github.com/CerebrasResearch/cb_optillm/tree/cepo), is licensed under the Apache License, Version 2.0 as detailed in [this LICENSE](./LICENSE)


Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

Initial change: 
Adding Cerebras Planning and Optimization (CePO). CePO is a method to empower Llama with reasoning, via test-time compute. [Learn more in the blog](https://cerebras.ai/blog/cepo)

At a high-level, in CePO, we make m attempts to generate n step-by-step plans, refine the plans, check inconsistencies between them, use the above feedback to generate the final plan and produce the answer. This process is them repeated N times in a classical best of n manner.

Ongoing Apache-licensed contributions:
* Added the implementation of CePO
* Integrated CePO with optillm
See updated files [here](https://github.com/codelion/optillm/compare/main...CerebrasResearch:cb_optillm:cepo)


Last updated: 01/14/2025
