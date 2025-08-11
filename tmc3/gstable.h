#ifndef PCC_3DGS_HELPER
#define PCC_3DGS_HELPER

#include <array>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

namespace pcc {
    extern const int GS_PROPERTY_SCALE;

    class NotImplementedException : public std::exception {
    public:
        const char* what() const noexcept override
        {
            return "Implementation is not exist";
        }
    };

    class GSHelper{
    public:
        GSHelper() = delete;
        ~GSHelper() = delete;
        GSHelper(const GSHelper&) = delete;
        GSHelper& operator=(const GSHelper&) = delete;

        static std::vector<std::string>& getGSKeys() {
            return gskeys;
        }
        static void addGSKey(std::string s){
            gskeys.push_back(s);
        }
        static std::string getGSKey(int idx){
            return gskeys[idx];
        }

        static size_t getGSKeysSize(){
            return gskeys.size();
        }

        static bool isExist(const std::string key){
            return std::find(gskeys.begin(),gskeys.end(),key) != gskeys.end();
        }

        static std::vector<std::string> gskeys;
    };
}

#endif