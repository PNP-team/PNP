#pragma once
#include <array>
#include <cstdint>

namespace at { 
namespace native {

template <>
struct NTTHyperParam<MNT4753_Fr_G1> {

    constexpr static const std::array<uint32_t, 24> group_gen = {0x7ff6635e, 0xeee0a5d3, 0xcfa1cff4, 0xff458536, 0xd8169ab0, 0x659af978, 0x4780e3f1, 0x1f1841c2, 0x6dcfef3a, 0x60221303, 0x9d72db20, 0xd1d5c8f3, 0xc0ffefab, 0xeb8b63c1, 0x5f6cfa4e, 0xd2488e98, 0x23f7a66a, 0xcce1c2a6, 0x5085b19a, 0x2a060f4d, 0x6408842f, 0xa9111a59, 0xd50bf627, 0x00011ca8};
    constexpr static const std::array<uint32_t, 24> group_gen_inverse = {0x1a5a51d7, 0xd247489e, 0xd30a598a, 0xa348baad, 0xe6165433, 0x2a92ab07, 0x665e9510, 0x55b20387, 0x279edbe0, 0xfe734039, 0x661b1fba, 0x983a06ee, 0x12782373, 0x4e4d4a96, 0x23a6dd11, 0x9e34b2de, 0x6b16d0f6, 0x10975064, 0xf112234e, 0x3707ca8d, 0xedc9dc96, 0xb32c5d32, 0xad5519cc, 0x00008c6b};

    constexpr static const std::size_t S = 30;

    constexpr static const std::array<uint32_t, (S + 1)*24> forward_roots_of_unity = {
        0xb193511b, 0xad2493c8, 0x3a5dc859, 0xe43fe801, 0x7a76f842, 0x320c18a5, 0xfa90b316, 0xe514f1c4, 0x1dfa22c7, 0x6241f537, 0x7ebc5916, 0x4ecea469, 0x2079d701, 0x3a541d0a, 0xff38c333, 0xe56e91d2, 0xc43d7bca, 0x9e7ae943, 0x47ae0c5c, 0xaddfabdf, 0xc0cd8c15, 0x17939b82, 0x07a611bc, 0x0000700b,
        0x7a8141ff, 0x0f0b49bc, 0xd986b0d7, 0x6b9d2b00, 0x596cfb35, 0xe6475c45, 0x135b6bae, 0x1a89022d, 0x7de3615d, 0x9715fce0, 0x9720c1cd, 0x1d5707cf, 0x87938075, 0xf7283703, 0x9b0cf951, 0xc37ad8ee, 0x22a6f621, 0xa2c3a2b1, 0xd8fa5ac7, 0x8288f804, 0xd145e3a8, 0xa77fd3a3, 0x459cd7fd, 0x0001406d,
        0x583d7b2d, 0xbb567bf4, 0xb454792b, 0x900327a6, 0x6f0716d7, 0xf8b50ffc, 0x2776c412, 0xb7ca0ce7, 0x2256fe7e, 0xa8c476d5, 0x7ad003fb, 0xb8b9cedf, 0x110ba1ce, 0x14d47228, 0x3f515a60, 0xc3bf0581, 0xad054974, 0xd53406f0, 0x265cde77, 0x3eda55b2, 0xc9440efb, 0x6432e6e8, 0xe87e40fe, 0x00004626,
        0x9d0ffa67, 0xc617de28, 0x2d4d18f0, 0xc65b89fa, 0x2651067d, 0xf948ea6c, 0x1dd18f20, 0x68fdea6a, 0x9bb015d3, 0xbe94595d, 0x7558a0f5, 0x7c922609, 0x3df42f16, 0x7959cfab, 0xe8e82796, 0x06c76818, 0x3bbf542b, 0xdd6a942f, 0xa5b3729d, 0x6243d3c1, 0x6bc60f18, 0xba1109e7, 0x78022ea3, 0x0001a2f6,
        0x849a35de, 0xa8a1b434, 0x47ef3f10, 0xcf9c7459, 0x7420b96d, 0x4a325c14, 0x4f4f22c6, 0x9f51e43a, 0x20c3dbcb, 0xf086ea0c, 0xadeb007c, 0x28cec437, 0xe300e1aa, 0x90d3efcb, 0x149a8bad, 0xe6bcf84c, 0x9646e75a, 0x5f0b9d5a, 0x06e98e34, 0x7e0fb29d, 0x7a43df52, 0x010b3299, 0xc8e21803, 0x0000b510,
        0xd251d751, 0x82135245, 0x60f0176e, 0x94e5f2f8, 0x2fd303dc, 0xfe3f4862, 0x6a9dafd6, 0xd8b0f113, 0xc1ea062b, 0x226affb2, 0x2421b2c2, 0x320e81c7, 0xf9978348, 0x78be0ad7, 0x54c7a00b, 0xd1a6c6ce, 0x7b39ca8f, 0xe4047c72, 0x3382cbf5, 0x720d195d, 0x8c753643, 0x5dd79c3e, 0x28e13503, 0x000002fe,
        0xeabc8894, 0x3bbc50f7, 0x91a63841, 0x1b1875a5, 0x3f11da20, 0x8bd6d613, 0x2d3b3caf, 0x85d21ada, 0xa9489b26, 0x379bb870, 0xb0d0bdf7, 0xb18a3ee1, 0xf7c2c1ee, 0xe28e8899, 0x6d4f546d, 0xf00d4e09, 0xe33f1572, 0x09a07bb1, 0x0f0ca2e2, 0x14c0872d, 0x7094131c, 0xc40fa2fa, 0x6f356f38, 0x00012ed3,
        0x81ec1e8b, 0xa07492ab, 0xcdef74f3, 0x40ac8713, 0xd4068316, 0x464f3fe7, 0x69417b08, 0x54e7e39d, 0x498811b0, 0x8567b9dd, 0x97ed1ee8, 0xa3045617, 0xe69e34f4, 0x6aa41770, 0x45de4730, 0x698fc656, 0xc2357692, 0x5ca35ec0, 0x21fd8747, 0xae3e6142, 0x265a86ef, 0x24167b0a, 0x41f13335, 0x0001a992,
        0x8495c54c, 0xc6f6bb46, 0xb3e11898, 0xe6b84189, 0xd23f329c, 0xda11ca5b, 0x47b2457d, 0xbf723483, 0x3bd75c20, 0xfb599cdc, 0x7dc1d47e, 0x128f45f9, 0x5bb52d56, 0x1845c0bb, 0x497659f8, 0xbf63815f, 0x86a504c9, 0xac20273c, 0x1e5af497, 0x6e99404c, 0xa8ebb9e7, 0x228c1002, 0x6c832287, 0x000069b1,
        0x8ace9c44, 0x05089473, 0xb1560e24, 0xf4569993, 0x12544041, 0x18662fad, 0x2f35a9b3, 0x72bce978, 0x25ed8f43, 0x2cfec796, 0x8e062ee3, 0x3f68edd2, 0x23ddcf4a, 0xcc38bfa1, 0xfc4d85e9, 0xb2ad3e65, 0xa9a22f71, 0x705dfe85, 0xd9efaaa7, 0x786a293d, 0x0e5897cd, 0x0d068549, 0x1917cfba, 0x000114af,
        0x91746ec8, 0xd4719a4d, 0xb69d7cf4, 0x78a26672, 0x92945541, 0xdf2da631, 0x9eaa0c33, 0xe4d9ab19, 0xab86d3de, 0x634414a9, 0x4c8b78a8, 0x6011934d, 0x28755bad, 0x20f6835a, 0x8e21b4d7, 0x6c305260, 0x79819e90, 0x5adaec0b, 0x6d4974a5, 0x49093641, 0x7df4c4a8, 0x64af0341, 0x513e68ac, 0x00010f9a,
        0x1921d71a, 0xd177600a, 0x9151fcfa, 0x541bdf95, 0x6bc0cfc3, 0xb9bcea10, 0x2051522c, 0x4d6702a8, 0x73083e5e, 0x7d77a8d8, 0x39b41f77, 0xf8bbc6af, 0x1b7048bd, 0xbb6fff4b, 0x74f5575b, 0xebd48b09, 0xffc7822b, 0x42784f49, 0xc03e3265, 0x0dee6d8f, 0x292e022e, 0xc9281100, 0x59797594, 0x0000335c,
        0x80b17507, 0x4df6dbf4, 0xa03cf239, 0x4428bb0f, 0x11c30f48, 0xc0b9ce86, 0x735dd722, 0x395b8851, 0x21db3aa3, 0x32d062ab, 0x4de9c9b1, 0xe85c425f, 0x01436b91, 0xd6432bee, 0x591eef31, 0x314a6b7c, 0x1c99bf14, 0x4e84e2c7, 0x68b334aa, 0xc5ee7567, 0x45add09f, 0xe27c0e49, 0x4bfaab41, 0x00005309,
        0x7fc236b4, 0xa04d4343, 0x5e2c1429, 0x6f88be4e, 0xe7f5e1bb, 0x2d65acb2, 0xe5e89b6c, 0xd2ef8aaa, 0x74d64767, 0x20d838c8, 0x6a30d852, 0x5a42474a, 0xf2ee4d2a, 0xe02b9d9d, 0xc2d7b66c, 0x5cee4986, 0xf84e531e, 0x05a46c03, 0x7c5bfc46, 0x55e191a7, 0x94a0b37c, 0x66a2a3b4, 0x67b53974, 0x0000121c,
        0x7d5e8883, 0x844d351f, 0x0c5e0de1, 0xc9a5265a, 0xb0fd9242, 0xc49fe94c, 0x210104d4, 0x0221ffeb, 0xaebcd4e8, 0x90b07c3e, 0xf4fcea10, 0x058ad38f, 0x4ea77ae6, 0x28ce6311, 0xcfb35cca, 0x75e503ac, 0x0ad691ba, 0xcbd901ff, 0xd36dd5b7, 0x26546002, 0x1d5fa5d5, 0xb5694e9d, 0xf3c645bc, 0x0001405a,
        0x33200de0, 0x2fa102ba, 0x58eb3a22, 0x1b6f6338, 0x13b679d6, 0x0275a13a, 0x4c07d83e, 0x16fb9f10, 0xe6ececc9, 0x34a60835, 0x88eb4eca, 0xa85792ea, 0x15edc296, 0xec8d11e8, 0x0eb702f3, 0x1cf31c53, 0x3e155f2c, 0xa667ee15, 0xb486f394, 0x5451836c, 0x22cf23b2, 0xe6f282d1, 0xfe15571a, 0x00012ee6,
        0x04085bb2, 0x20e3e692, 0xb1b5ae90, 0x1ab419d9, 0x49d65a99, 0xdbecf8f9, 0x18c59a82, 0x2584e512, 0xa4cccf32, 0x136003a8, 0xaa3b84d7, 0xe4006fd7, 0xd481cdbc, 0xced092f1, 0xb5e27f14, 0xc6e680d5, 0xe66a62de, 0x3a939e52, 0x8dafc377, 0x516f9015, 0xde1b6dbe, 0x43a0e6a3, 0xfc4262cc, 0x00007b0c,
        0xe900c79d, 0xf55cde1f, 0x6862668d, 0xb7bc02b3, 0x168ea9d0, 0x933f4685, 0xdfa9bab0, 0x3de422b8, 0x2a57265d, 0x161a6002, 0x1f10071d, 0x3a14b580, 0x0eebd7e2, 0x614387f6, 0xdfb837e0, 0xb941141b, 0x72dee0e3, 0x0a310188, 0xd78a961b, 0x38799af4, 0x3afe97c7, 0x2309dd2e, 0x94443227, 0x0001b785,
        0x3d0357c9, 0x8f22bd99, 0x087ab0f6, 0x8ed867e9, 0xa9c5e513, 0x4d7e3923, 0x82d3958d, 0x4b27e046, 0xb81df604, 0x313fba56, 0x74bbc184, 0x0e81dd6d, 0xc54c61a8, 0x390e07d1, 0xaff49e3f, 0x8e816b25, 0x5832b2f0, 0xccaf57c8, 0x73bcbd99, 0xc47af0d5, 0x543bf4df, 0x90e9fdf1, 0x7078dc92, 0x0000fb88,
        0xa793629c, 0x1c3331f1, 0x220b7f38, 0xc65c76ca, 0xad852585, 0xa861fdb0, 0xfab8a1ea, 0xb5119da9, 0xa2d42396, 0x61d263c9, 0x8f2064a9, 0x2180d7e0, 0x96eade6a, 0x7c57f340, 0x8cac3f75, 0x45c1dd40, 0x8bb740f8, 0xb2ad0ccd, 0xd2fac3d9, 0xdcb62b99, 0xb12a09fa, 0x63338502, 0xe77c1280, 0x00013e40,
        0x5654f209, 0xca5eb968, 0xeb5cd8be, 0x8f3d3717, 0xa8e80c15, 0x8cf948f9, 0xc796fff5, 0xb67f2cd9, 0x1e9157ce, 0x656051d9, 0x5911cff9, 0x5b921c28, 0x0a92e2e6, 0xcbdb875a, 0x7692ed19, 0x021a622a, 0xb3e4c0df, 0x784150a9, 0xc2e6ad6b, 0x20075c18, 0x1106f444, 0x1356b029, 0x92eb8f5d, 0x0001bd7f,
        0xaf768180, 0x836a979a, 0xa0802272, 0x98122047, 0xfd7e5dcb, 0x539f7f03, 0xb291d13e, 0xd456eaeb, 0xccb7a059, 0x184c93dd, 0x4e80f93a, 0x5451f576, 0x91a4c773, 0xddf03128, 0x3cafc680, 0x00d9c859, 0x6c7682cd, 0x045feac6, 0x840cb59a, 0x60548570, 0x1bbb1698, 0x58ab88ef, 0x67afb1f4, 0x00003831,
        0xfda2fa48, 0xef060da7, 0x791d47c4, 0x031345a3, 0xe453d25c, 0x4b966d79, 0x550644b3, 0xc613fcac, 0xc1e50a3a, 0xdcdc7582, 0xcb62a1d5, 0xdcb0ce10, 0xe3a90071, 0xa3d7572f, 0xb37efa97, 0x682357b7, 0x5d4398af, 0x60bea1d1, 0x609c4336, 0x9435870f, 0x42f2f57b, 0xe37c2f0f, 0x9590edc8, 0x0000121c,
        0x9fe6f496, 0xd8c60d66, 0xeda242aa, 0xaf3f7852, 0xbbc300be, 0x5b3867a7, 0x46ecfc25, 0x3eb90bbb, 0xd7bc274b, 0x72a35d4c, 0x6992c65a, 0x888e8978, 0x2751aa46, 0x7fb16404, 0x53054272, 0xb94334cd, 0x846517a1, 0x7f319b00, 0x49663be5, 0xbaac6daf, 0xf017dac9, 0x7f19e381, 0x5a3a75ad, 0x00017d47,
        0xd27a32c7, 0xd221282e, 0x58a1ad31, 0xb16d661c, 0xb1287234, 0xe8e8dca2, 0x4a2873c2, 0xcdf6bc8c, 0xfb169bf1, 0x31255751, 0x1b058404, 0x7ae318aa, 0x879352cd, 0x2d3441bc, 0x23f9bd18, 0x39c7fe3f, 0x23554b47, 0xb14173f1, 0x8545803a, 0x7fe7ec60, 0x4f749064, 0x6c9c8d2c, 0xb3222b67, 0x000137b1,
        0x75250fe2, 0x9debb176, 0x64682198, 0x05466f99, 0x35633460, 0x54ebd23f, 0xf5c81391, 0x575ad502, 0x9c80af7a, 0x871aa80e, 0xe6ed5989, 0x439acca5, 0xb32e2861, 0xd08befcd, 0xacf0ef46, 0xb7ab0572, 0x1a15574c, 0x7e26f754, 0x39de48ae, 0xd4c20c08, 0x90a056ec, 0xca2d33e0, 0x7b2f30db, 0x0000d06b,
        0xf0a3b27a, 0xf7e9a3a6, 0x1dcaa1f4, 0x3c14932e, 0x1a253df2, 0xc27cd159, 0xb0d5fd8c, 0xd4e79b90, 0xbfabea75, 0xa136a6c2, 0x1634cb45, 0x02316c06, 0x37b33f9d, 0x9dab5600, 0xc71bb9c7, 0x29bf328f, 0x510b6079, 0x83232437, 0x576415fb, 0xff05c57f, 0x27953ae8, 0x40f97eb2, 0x29d34d50, 0x00002206,
        0x00c346f1, 0x13096a8b, 0x35d4c9d1, 0x67b379ed, 0xaede71aa, 0xb708a509, 0x3876d570, 0x5ed15224, 0x2d00dfac, 0x60669461, 0x06422725, 0x14f3aed0, 0xaa25f05f, 0x69e23523, 0x9e610835, 0xf5e6c1a3, 0x5981b923, 0x66e345f3, 0x6652ac06, 0xbea17032, 0x8d343302, 0x2cdbbc04, 0xf79c1993, 0x0001950b,
        0x8a0af947, 0x4d8f1b4f, 0x58277616, 0x641b4c7c, 0x9a9d0379, 0x9f66ae91, 0x018fec38, 0x41e68adc, 0x9f8cd2b5, 0x01a1ebfd, 0xb77be256, 0x086a0fda, 0x1bb216a3, 0x07b54350, 0xbf4846b5, 0x0c1e7da9, 0xc2b746a4, 0x03937993, 0x3076661e, 0x4db7acb9, 0x61fde6cc, 0x91973014, 0x3ce2b2ba, 0x00000fae,
        0x73260782, 0xdc2b3197, 0x74c86452, 0xa3d89b14, 0xd5883bd2, 0xb3043d0a, 0x051a48ba, 0xa85c962b, 0x254c27db, 0xffdee673, 0xa95a7d8d, 0x7f397495, 0xafaeec01, 0x7a174d70, 0x1957a31b, 0xa0b31515, 0x770ff99e, 0x714e59f7, 0x905bf863, 0x5dc04fb3, 0xc9a44380, 0x7d6dad99, 0x44d12c1a, 0x00000d8e,
        0x97671883, 0x307f66b2, 0x1e645f4e, 0xd72a7f2b, 0x9a902283, 0x67079daa, 0xa86c668b, 0xf33f7620, 0x66464c12, 0x8878570d, 0x524f522b, 0xa557af5b, 0xef19319d, 0x5fafa3f6, 0x10a65629, 0x1eb9e041, 0xc639a0b0, 0x3f96feb3, 0xf3ffd732, 0x4d4fe37d, 0x55bcf3e9, 0xadc831bd, 0x2a8bd6ab, 0x0001b9f3
    };

    constexpr static const std::array<uint32_t, (S + 1)*24> inverse_roots_of_unity = {
        0xb193511b, 0xad2493c8, 0x3a5dc859, 0xe43fe801, 0x7a76f842, 0x320c18a5, 0xfa90b316, 0xe514f1c4, 0x1dfa22c7, 0x6241f537, 0x7ebc5916, 0x4ecea469, 0x2079d701, 0x3a541d0a, 0xff38c333, 0xe56e91d2, 0xc43d7bca, 0x9e7ae943, 0x47ae0c5c, 0xaddfabdf, 0xc0cd8c15, 0x17939b82, 0x07a611bc, 0x0000700b,
        0x7a8141ff, 0x0f0b49bc, 0xd986b0d7, 0x6b9d2b00, 0x596cfb35, 0xe6475c45, 0x135b6bae, 0x1a89022d, 0x7de3615d, 0x9715fce0, 0x9720c1cd, 0x1d5707cf, 0x87938075, 0xf7283703, 0x9b0cf951, 0xc37ad8ee, 0x22a6f621, 0xa2c3a2b1, 0xd8fa5ac7, 0x8288f804, 0xd145e3a8, 0xa77fd3a3, 0x459cd7fd, 0x0001406d,
        0xe7c284d4, 0x1db0faed, 0x5b4cc124, 0xbe9d7170, 0xcff940bf, 0xde0e71bf, 0x0d227691, 0x0215ec8f, 0x06ca27b8, 0x962753bf, 0x4d89a59f, 0xf9b28d49, 0x904f55ce, 0x84fcb2b1, 0xa94f932d, 0x443eb3a4, 0xbf928efe, 0x8983e208, 0x3532d175, 0x791f419e, 0x259ebeb2, 0xabefa93a, 0x45148312, 0x00017e9f,
        0xc64fbf31, 0xf69cc937, 0xb77ce5ff, 0x4b8e2cce, 0x83f4f57a, 0xac401874, 0x06524a08, 0x9a128f65, 0xff87993f, 0xba539967, 0x77455cd8, 0x26def26d, 0x51c99ee9, 0x3e55ec52, 0xefff98d7, 0xaa7566a7, 0xe3ca1574, 0xe79e3885, 0xcb25f04e, 0x32d3b6b6, 0x4121e583, 0x365edc8b, 0x57d8e1d2, 0x000196a2,
        0x00b14c73, 0x776d1a57, 0x08791618, 0xaf0869f0, 0x662d7340, 0x49fe47bc, 0x7606ded7, 0xf91318c0, 0x0052bcf0, 0x767edaa3, 0xe4d9e6d0, 0xed4ccba2, 0xbbcf1217, 0xb8172046, 0x7e384a9f, 0xda97fadc, 0xa7db4acf, 0x26269723, 0xe5f2a1d7, 0x0e66e434, 0xee96bdb2, 0x8c256531, 0x212bf837, 0x0001a2f5,
        0xe68d0a02, 0x7a2b9b2b, 0xe6817159, 0xa785903b, 0x82673657, 0xdba44f04, 0x39b24864, 0x2bf8b500, 0x6186ee2f, 0x06b0415d, 0x5e39741a, 0x62149a25, 0x0bbf2960, 0xf7458f2d, 0x52047ae6, 0xdabebfe0, 0x2c69ebf4, 0xcafbdd40, 0x68017ea1, 0x0f7c6bd7, 0xb072ef67, 0x39065be2, 0x9ed2a159, 0x0001909d,
        0xd4e548aa, 0x8b1a48bd, 0xe941f72f, 0x9feacb2d, 0x1699205b, 0x412f2380, 0xe7facf84, 0x848986a5, 0xe4c4df37, 0x301c0708, 0x0b452d06, 0x4e1b73c4, 0xeadc4b35, 0x3aa86ce0, 0xb3280eaa, 0xcd61df26, 0x6943ea2b, 0x9787f750, 0xc4ddb064, 0xe5b10ad2, 0xbddbb57d, 0x669acefd, 0x13704ce0, 0x0001b4f0,
        0x00d42c55, 0x3eee97dd, 0x24e336a5, 0x6c35ca09, 0x9ccbe61e, 0xedaed260, 0x2d128750, 0x7df0ff2f, 0xdcc467af, 0xa1d12220, 0xa18b8a42, 0xdbecb709, 0xa2358697, 0xb8587b78, 0x4a2471cf, 0xee5b38a5, 0x4d2ccef2, 0x83f1b20a, 0xa7ec698e, 0x4d54b442, 0x605e0d3f, 0x92a53b4f, 0x8f20daa1, 0x00017fe3,
        0x0aaf98d7, 0x66ccf630, 0xa10857d8, 0x188881d7, 0xb905cd6f, 0x3c5a3766, 0x5b1ace93, 0x13dd1a65, 0xf251f191, 0x2957353f, 0x09970db1, 0x2b52595c, 0x0b069a08, 0xffbb57e2, 0x0c3b5006, 0x1b7a5b78, 0xab71b1eb, 0xb8679f3e, 0x99e6ec95, 0x2c985d57, 0x2530f237, 0x7d1f5400, 0xee4b5639, 0x0000da6c,
        0xada1a1b6, 0xb4480dc8, 0x4ce9c6de, 0x937e31e1, 0xa3b616d2, 0x1a4cd9d2, 0xe391b681, 0xfab1236f, 0x90386ed4, 0xf345f1e9, 0x582942b0, 0x3843cef8, 0x3e5084d9, 0x296e3b2a, 0xd155ae19, 0x5500109f, 0x03ee44a4, 0x31ca5529, 0x798e4c74, 0x1622e7f2, 0x2726b3aa, 0x72750925, 0x31bbc38a, 0x00011d8b,
        0x029d78c3, 0x310a0844, 0x2081620f, 0x73fd5d5b, 0x218b58f3, 0xf917990f, 0x801b37c1, 0xf1b20057, 0x04e3d3f6, 0xc10f3128, 0x788477be, 0x4cb13ccc, 0xa4887462, 0x30a682df, 0x2671ccb6, 0x22525636, 0x18981d33, 0xeecb2edf, 0x2094dd48, 0x0bd5ca22, 0x1024b119, 0xff24eb74, 0x3df3a0f9, 0x0001b227,
        0x6d8c2626, 0xefa8bc1a, 0x0e78ce91, 0x3c6a6037, 0x7a59cbd1, 0xcdf05c85, 0x6d7b47d7, 0x67a4a4d8, 0x82bfacb6, 0x7fbb65ff, 0xa90ffb31, 0x8492d5f4, 0xd0fb7170, 0xc8b6eceb, 0x1f050a8a, 0xc3d03296, 0x356fe4db, 0x7fcb112e, 0x8412337a, 0x4b29076d, 0x14bb5a8d, 0x62fadc0c, 0xddcee387, 0x0000eb8d,
        0x26d091bc, 0x1d94b69f, 0x0f46688e, 0xadf8d72b, 0x47852ff9, 0x8755e24a, 0x5ff09c6a, 0xf7981c70, 0x5f2278a7, 0x4be9168c, 0x87b1c430, 0xadb0b26f, 0x7087d74f, 0xe77da8c5, 0xfc3bb5d7, 0x5996d442, 0xa9d95c4e, 0x17621abc, 0x8281468f, 0xb2e45ee3, 0x82ad9d89, 0x17c0477f, 0xe2b27be5, 0x0001055e,
        0x9a929739, 0x48b7d53b, 0x86bc514b, 0xcd1d60ba, 0xb8c5560c, 0x1ad570b9, 0xb181b84d, 0xa20eb552, 0x2ef19cbc, 0x406d065e, 0xdc4403c2, 0xa6fcb571, 0x9cb33cb5, 0x75b0fe44, 0x0ccb294e, 0xb6a4cfdc, 0x3185885b, 0x74c7e63b, 0x4d9cba39, 0x4877e9bb, 0x0b7eddf7, 0x7dc84c0d, 0x1dd782eb, 0x0000e515,
        0xe1b20659, 0x2bbf2d75, 0xb31ae283, 0x302020dd, 0xc6370c2a, 0x1f3fea91, 0xe946bab0, 0x2f15ce3c, 0x03278b8d, 0x47300488, 0xb7c98977, 0xb6dacddf, 0x4c6cf60f, 0x44ef69c1, 0xd73f6475, 0xa111bc3c, 0x31c50b87, 0xd3ea2dfd, 0x13aa52b4, 0x45ec407c, 0xcf14f185, 0x520b5e0d, 0xed77e106, 0x0001b379,
        0x7ce9d438, 0x83d4cd7b, 0x888b6371, 0x2bb907aa, 0x175f8bd2, 0xc4d7848e, 0x4d49da6c, 0x7c23e6ac, 0xf105ce61, 0x5a84ad31, 0xc2c4497e, 0x8d2abef4, 0x8779c139, 0xe4fe137c, 0x377aab71, 0x763b5653, 0x156fbc7e, 0x1aac3528, 0x957eed51, 0x4dd17697, 0x7947eeb2, 0xdad7c162, 0xaded8ae7, 0x0000579a,
        0xdaaa5fda, 0xbcf08c86, 0x9bdb267b, 0xcee10e67, 0x28eab4d7, 0xa9830c7d, 0xd8448842, 0xf942d9ea, 0x3f64ff47, 0xc83dfb48, 0xff267e00, 0xff3380f8, 0xfdd598d5, 0x6669a734, 0x15fef913, 0x5f6664bc, 0x9f1a6513, 0x90745bf4, 0xb35681c8, 0x4177de1f, 0x3918d5c2, 0x70c458be, 0x84d6298f, 0x0000121d,
        0xa17eafab, 0x4c4c2aaa, 0x4446a0dd, 0xfde0e91d, 0xb4781bdd, 0xf3744d57, 0xa152878d, 0x4e0f16d4, 0x15871850, 0x1daf75ec, 0xa140ac1b, 0x12dd26b6, 0x573bec89, 0x424de0bc, 0x437909cf, 0x2c48627a, 0x6f1fc05e, 0xa495bcfe, 0x48c8ca0f, 0xe5efde83, 0xfe9c2879, 0x12e76626, 0x8b4013e1, 0x00012127,
        0xb4fc7935, 0x5e08311b, 0x9bdc98c0, 0x5bb6ff63, 0x53ff54fb, 0x6756bcc0, 0x91984055, 0xd9f53693, 0x03423808, 0x6eb0d870, 0xfa4c451f, 0x7371c3c7, 0x133e61c1, 0x4525e7cf, 0xf551fe9f, 0xdb0863c5, 0xe80b2c2f, 0x1e95ceb8, 0x7d9db748, 0x79eb7682, 0x678885ee, 0xd806977e, 0x97397bcc, 0x000106e6,
        0x51be9e5a, 0xf6eba95f, 0x361632e3, 0xc4e46981, 0x339977cf, 0x59f4dd29, 0xcd980d86, 0x63dc7b2c, 0xdb2f4258, 0xf41015a8, 0xb6fdea41, 0x810d74f0, 0xcb84bce4, 0x1e3f99d6, 0x240b65f9, 0xe4d88a58, 0xfd7da92d, 0xd78aad47, 0xb82ef4ae, 0xceb6660d, 0xabfb9dce, 0xbf90d41e, 0x8efb1c98, 0x0000f88e,
        0x7069e7bb, 0xf41faeca, 0xb1541bd4, 0xd1bfb814, 0x504d363e, 0xd0bbc3ac, 0xa94eb7eb, 0x7bf76fae, 0x1899272b, 0xedf94b30, 0xe5b24410, 0x146500ae, 0x796a4c68, 0x19383766, 0x74412573, 0x35803cc7, 0xca84aa31, 0x14fa4738, 0x89599055, 0x67d73ee5, 0xec3d7266, 0xc8e5fdfc, 0x0f1586ed, 0x00003d86,
        0x384faa3f, 0x57280488, 0xda41815f, 0x48b9a85d, 0x00062613, 0x09bf2424, 0xeb136fc0, 0xb2a0af7b, 0x385a8b6d, 0x07dc14b3, 0xefe18d48, 0xa91da706, 0xa93c6512, 0xef248a01, 0x2b03dec8, 0x26e8b5af, 0x6317f5e7, 0xca397ace, 0xb714a283, 0xf13e0e93, 0xe565a6b2, 0x7e52ea1b, 0xbff0af96, 0x0000a53f,
        0xbc58dab1, 0x5ccd6b28, 0xc36757c5, 0x15734a25, 0x2fcac33b, 0xc991d2b2, 0xfeaab792, 0x78cc06e8, 0x02dc2e61, 0x627aebcf, 0x90351000, 0x9e0f5541, 0x11d8ced9, 0x9d80f0f7, 0xd394ab40, 0x7e0ebd37, 0x298438ea, 0xaff00cc4, 0x3aa2f9be, 0xb4f92323, 0x3efe1e41, 0xed3d26aa, 0xe89afc6c, 0x0000cbbb,
        0x05f87ab5, 0x6abf746f, 0x4444b38e, 0x2deb8cd7, 0x6767a02d, 0xc8b566c3, 0x13c6cd6a, 0x262bd5f1, 0xfd28c9c8, 0xc8df69b2, 0x0f56cb99, 0xdd660402, 0x4eb4d994, 0xb6881afd, 0x6fae5eba, 0x55366adb, 0xc837d03f, 0xcecd0d96, 0x113e0415, 0x4b1e1ee0, 0xbf98b961, 0xe51e1205, 0xe2fb8074, 0x000038e4,
        0x9edf0492, 0x0e7dd6c1, 0x16733351, 0x0df92933, 0x1113d88d, 0x00a7ccca, 0x143bba4f, 0x7674c8c7, 0x34c0c254, 0x0fde200f, 0x27eb56ab, 0xcab34b12, 0x30cfd52c, 0x2deae2ce, 0x1eb2595f, 0x18fb8c81, 0x46c1e2f0, 0x5d5790af, 0xc3487059, 0x70556eee, 0x8949783f, 0x982e4340, 0x0c265353, 0x00013097,
        0x8dbd1f1b, 0x3327a8b7, 0xb32448c3, 0x4315a6c1, 0x7457e9cf, 0xb6d37af7, 0xd7fa1187, 0xb988b289, 0x87d24637, 0xe2d1bffd, 0x6438da6b, 0x8a06214a, 0x68cffae9, 0xea63f46d, 0x7041a8da, 0xc620ff65, 0x5917810c, 0x8f9111b6, 0x980bfb22, 0xb61e8b68, 0x2bd6dac6, 0xb56e3c2c, 0x0bb8388b, 0x0001a6fb,
        0x668caf74, 0x6ca647e0, 0x44ad908c, 0xb4127cf9, 0x750fc8a2, 0x7541d60f, 0x25bf8eb3, 0x0474ed90, 0xc8179e0e, 0x257ea0b5, 0xc54426ed, 0x26c52ddb, 0xc8e28c23, 0x9dcfff8d, 0xe6b7d5e0, 0xcc010814, 0xbc546a4e, 0xed2ecf0e, 0x9cfef052, 0x7671c592, 0x25b1a922, 0x64a6b742, 0x917d897c, 0x00015dcc,
        0x536df42a, 0x7bf40289, 0xf0e943c4, 0x655a9bda, 0x310f135f, 0x6cda10b4, 0x010d7e38, 0x9e0d0009, 0x397fe606, 0xf05c9334, 0xa509ccfd, 0x1171e3b3, 0x93505a5f, 0xed2a44fa, 0x24941d4c, 0x8d622335, 0x05937e71, 0x17d24013, 0x9480c29e, 0xcc1879ca, 0x74a79bc1, 0x52b1dfbd, 0x4e2c0d8d, 0x0000ba1e,
        0x616a037f, 0x594baf65, 0x2ac3f62d, 0xa298cc29, 0xaf54d753, 0x48e2ef51, 0x2e8b22e1, 0x6cea5b96, 0x96f13ea1, 0xf491d5a4, 0x467c3ae5, 0x273c5816, 0x6fff0a7f, 0x9bacb867, 0xccb924c5, 0x8d849124, 0x6d44e968, 0x77365207, 0xf579b7d9, 0x67a3db91, 0xc0d09bb3, 0xaf274c4b, 0x1796abea, 0x0000b41a,
        0x1b1a49f0, 0x677c0ff5, 0x1d938d12, 0xfcb29f29, 0x85d6acf2, 0x8dbaf38d, 0x75cd4b8f, 0x902bc8ac, 0x4be7d6d5, 0x6a5f7c1f, 0x1bf28553, 0x76874b0c, 0x8202b1ac, 0xe6098a89, 0xf1531ed0, 0xbda67e26, 0x9b3f2f4e, 0x4fcc7d7a, 0x8e8ee53b, 0x0bfaef36, 0xc1fdda22, 0x05efa1fa, 0x2c05e694, 0x00003166,
        0x4384704c, 0x82065276, 0x6804148b, 0x7bd614f2, 0xc15c72c4, 0xd74f5d61, 0x98e6f11a, 0x115b9f20, 0x0c01cdfc, 0xf12dfb06, 0xb5628ed5, 0xecb9593d, 0xf34b3672, 0xefef0e23, 0xdc16f746, 0xd7d899ec, 0x08aa6dc8, 0xf6794339, 0x24850fbf, 0x52643b61, 0x5435c73e, 0x6770b4c5, 0x4a451851, 0x0000e4de
    };

    constexpr static const std::array<uint32_t, (S + 1)*24> domain_size_inverse = {
        0x7fff6f42, 0xb9968014, 0xb589cea8, 0x4eb16817, 0x0c79e179, 0xa1ebd2d9, 0xc549c0da, 0x0f725cae, 0xd3e6dad4, 0xab0c4ee6, 0xde0ccb62, 0x9fbca908, 0x13338498, 0x320c3bb7, 0xd2f00a62, 0x598b4302, 0xfd8ca621, 0x4074c9cb, 0x3865e88c, 0x0fa47edb, 0x1ff9a195, 0x95455fb3, 0x9ec8e242, 0x00007b47,
        0x3fffb7a1, 0x5ccb400a, 0xdac4e754, 0xa758b40b, 0x863cf0bc, 0x50f5e96c, 0x62a4e06d, 0x07b92e57, 0x69f36d6a, 0x55862773, 0x6f0665b1, 0x4fde5484, 0x8999c24c, 0x19061ddb, 0x69780531, 0xacc5a181, 0xfec65310, 0x203a64e5, 0x9c32f446, 0x87d23f6d, 0x8ffcd0ca, 0x4aa2afd9, 0xcf647121, 0x00003da3,
        0x3fffdbd1, 0x1ae95b76, 0x753310d2, 0xfafca691, 0x629ea429, 0x13dcb594, 0xcb9f0d89, 0x60cc93e6, 0xc98a49d0, 0x4a38f903, 0x9bb007a6, 0x01255856, 0x957a5cf5, 0x596ba15a, 0xa90c795f, 0xda61ad53, 0xb5af15c1, 0xbf7926ef, 0xfbe15219, 0x1fe5eb5e, 0x3f6fcf3c, 0x2d629ffe, 0xfe7b9a99, 0x00010134,
        0x3fffede9, 0xf9f8692c, 0x426a2590, 0xa4ce9fd4, 0x50cf7de0, 0xf5501ba8, 0x801c2416, 0x8d5646ae, 0xf955b803, 0xc49261cb, 0xb204d8a0, 0x59c8da3f, 0x1b6aaa49, 0x799e631a, 0xc8d6b376, 0x712fb33c, 0x9123771a, 0x8f1887f4, 0xabb88103, 0xebefc157, 0x97294e74, 0x1ec29810, 0x96072f55, 0x000162fd,
        0x3ffff6f5, 0x697ff007, 0xa905aff0, 0xf9b79c75, 0x47e7eabb, 0xe609ceb2, 0x5a5aaf5d, 0x239b2012, 0x113b6f1d, 0x01bf1630, 0x3d2f411e, 0x861a9b34, 0xde62d0f3, 0x09b7c3f9, 0x58bbd082, 0xbc96b631, 0xfedda7c6, 0x76e83876, 0x03a41878, 0x51f4ac54, 0xc3060e11, 0x17729419, 0xe1ccf9b3, 0x000193e1,
        0xbffffb7b, 0x2143b374, 0x5c537520, 0xa42c1ac6, 0x43742129, 0x5e66a837, 0x4779f501, 0xeebd8cc4, 0x1d2e4aa9, 0xa0557062, 0x82c4755c, 0x9c437bae, 0xbfdee448, 0xd1c47469, 0xa0ae5f07, 0xe24a37ab, 0x35bac01c, 0xead010b8, 0x2f99e432, 0x84f721d2, 0x58f46ddf, 0x13ca921e, 0x07afdee2, 0x0001ac54,
        0x7ffffdbe, 0xfd25952b, 0xb5fa57b7, 0x796659ee, 0xc13a3c60, 0x1a9514f9, 0x3e0997d3, 0x544ec31d, 0x2327b870, 0xefa09d7b, 0xa58f0f7b, 0x2757ebeb, 0xb09cedf3, 0xb5cacca1, 0xc4a7a64a, 0xf523f868, 0xd1294c47, 0x24c3fcd8, 0x4594ca10, 0x9e785c91, 0xa3eb9dc6, 0x91f69120, 0x1aa15179, 0x0001b88d,
        0xbffffedf, 0xfe92ca95, 0x5afd2bdb, 0x3cb32cf7, 0xe09d1e30, 0x8d4a8a7c, 0x9f04cbe9, 0x2a27618e, 0x9193dc38, 0xf7d04ebd, 0xd2c787bd, 0x93abf5f5, 0xd84e76f9, 0x5ae56650, 0x6253d325, 0xfa91fc34, 0x6894a623, 0x1261fe6c, 0xa2ca6508, 0x4f3c2e48, 0x51f5cee3, 0xc8fb4890, 0x8d50a8bc, 0x0000dc46,
        0xffffff70, 0xebcd20bb, 0x354f3315, 0xc5a9e307, 0x8fcebae3, 0x3207061c, 0x69cf0347, 0x7203ad82, 0xdd5a8137, 0x9b5e0ca8, 0x4d9098ac, 0xa30c290f, 0x3cd4b74b, 0x7a5b4595, 0x257a6059, 0x8147daad, 0xea963f4b, 0xb88cf3b2, 0x7f2d0a7a, 0x839ae2cc, 0xa06c4e48, 0xec8eec59, 0x5d71b666, 0x00015086,
        0xffffffb8, 0xf5e6905d, 0x9aa7998a, 0xe2d4f183, 0x47e75d71, 0x9903830e, 0x34e781a3, 0xb901d6c1, 0x6ead409b, 0x4daf0654, 0xa6c84c56, 0xd1861487, 0x9e6a5ba5, 0xbd2da2ca, 0x92bd302c, 0xc0a3ed56, 0x754b1fa5, 0x5c4679d9, 0x3f96853d, 0x41cd7166, 0xd0362724, 0x7647762c, 0x2eb8db33, 0x0000a843,
        0xffffffdc, 0x7af3482e, 0xcd53ccc5, 0xf16a78c1, 0x23f3aeb8, 0xcc81c187, 0x9a73c0d1, 0xdc80eb60, 0x3756a04d, 0x26d7832a, 0xd364262b, 0xe8c30a43, 0x4f352dd2, 0x5e96d165, 0x495e9816, 0xe051f6ab, 0xbaa58fd2, 0xae233cec, 0x1fcb429e, 0x20e6b8b3, 0x681b1392, 0xbb23bb16, 0x975c6d99, 0x00005421,
        0x7fffffee, 0xbd79a417, 0xe6a9e662, 0x78b53c60, 0x91f9d75c, 0xe640e0c3, 0x4d39e068, 0xee4075b0, 0x1bab5026, 0x936bc195, 0xe9b21315, 0x74618521, 0xa79a96e9, 0x2f4b68b2, 0xa4af4c0b, 0x7028fb55, 0x5d52c7e9, 0x57119e76, 0x8fe5a14f, 0x10735c59, 0x340d89c9, 0xdd91dd8b, 0xcbae36cc, 0x00002a10,
        0xbffffff7, 0x5ebcd20b, 0x7354f331, 0x3c5a9e30, 0xc8fcebae, 0x73207061, 0x269cf034, 0x77203ad8, 0x8dd5a813, 0xc9b5e0ca, 0xf4d9098a, 0xba30c290, 0x53cd4b74, 0x97a5b459, 0xd257a605, 0xb8147daa, 0x2ea963f4, 0xab88cf3b, 0xc7f2d0a7, 0x8839ae2c, 0x9a06c4e4, 0x6ec8eec5, 0x65d71b66, 0x00001508,
        0xfffffffc, 0x9be22476, 0xc17b16c0, 0xc57d9ba3, 0x03fea1a2, 0xa4f1f90f, 0x2d9b156c, 0x18801a27, 0x5b7b6725, 0x0450d5af, 0xde995993, 0x364e8f5c, 0x7a942189, 0x98bb6c99, 0x5d7c49c9, 0xe0091b68, 0x4da09e33, 0x85205c1a, 0x91c1404a, 0x2019a2be, 0x4474c949, 0xbf75bf74, 0x49b4efbb, 0x0000ece7,
        0x7ffffffe, 0x4df1123b, 0xe0bd8b60, 0x62becdd1, 0x81ff50d1, 0x5278fc87, 0x96cd8ab6, 0x8c400d13, 0xadbdb392, 0x82286ad7, 0x6f4cacc9, 0x9b2747ae, 0xbd4a10c4, 0xcc5db64c, 0x2ebe24e4, 0xf0048db4, 0x26d04f19, 0x42902e0d, 0x48e0a025, 0x900cd15f, 0x223a64a4, 0xdfbadfba, 0xa4da77dd, 0x00007673,
        0xbfffffff, 0x26f8891d, 0xf05ec5b0, 0xb15f66e8, 0xc0ffa868, 0x293c7e43, 0xcb66c55b, 0x46200689, 0xd6ded9c9, 0xc114356b, 0x37a65664, 0x4d93a3d7, 0x5ea50862, 0x662edb26, 0x175f1272, 0xf80246da, 0x9368278c, 0xa1481706, 0xa4705012, 0x480668af, 0x111d3252, 0xefdd6fdd, 0xd26d3bee, 0x00003b39,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00010000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00008000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00004000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00002000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00001000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000800,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000400,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000200,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000100,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000080,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000040,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000020,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000010,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000008,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000004
    };
};

}//namespace native
}//namespace at
