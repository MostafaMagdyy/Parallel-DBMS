#include <iostream>
#include <chrono>
#include <cstring>
#include <string>
#include <cstdint>
#include <bitset>
using namespace std;

__uint128_t string_to_uint128(const std::string &str) {
    __uint128_t result = 0;
    for (int i = 0; i < std::min((int)str.size(), 16); ++i) {
        result |= (__uint128_t)(uint8_t)str[i] << ((15 - i) * 8);
    }
    return result;
}
std::string uint128_to_string(__uint128_t value) {
    std::string result;
    for (int i = 0; i < 16; ++i) {
        uint8_t byte = (value >> ((15 - i) * 8)) & 0xFF;
        if (byte == 0) break;
        result += static_cast<char>(byte);
    }
    return result;
}


int main() {
    string s= "Mostafa Magdy";
    __uint128_t res=string_to_uint128(s);
    cout<<uint128_to_string(res)<<endl;
    
    
}