"""
SVTRv2Mamba 테스트 스크립트

필수 빌드:
    cd reference/DAMamba/classification/models/ops_dcnv3
    python setup.py build_ext --inplace
    
    cd reference/DAMamba/classification/models/selective_scan
    python setup.py build_ext --inplace
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch


def test_import():
    """Import 테스트"""
    print("=" * 60)
    print("1. Import 테스트")
    print("=" * 60)
    
    from openrec.modeling.encoders.svtrv2_mamba import SVTRv2Mamba, DASSM, DASSMBlock
    print("  ✅ SVTRv2Mamba, DASSM, DASSMBlock import 성공")
    return True


def test_dassm():
    """DASSM 단독 테스트"""
    print("\n" + "=" * 60)
    print("2. DASSM 단독 테스트")
    print("=" * 60)
    
    from openrec.modeling.encoders.svtrv2_mamba import DASSM
    
    B, C, H, W = 2, 64, 8, 32
    x = torch.randn(B, C, H, W).cuda()
    
    dassm = DASSM(d_model=C, head_dim=16).cuda()
    out = dassm(x)
    
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert x.shape == out.shape, "Shape mismatch!"
    print("  ✅ DASSM 테스트 통과")
    return True


def test_dassm_block():
    """DASSMBlock 테스트"""
    print("\n" + "=" * 60)
    print("3. DASSMBlock 테스트")
    print("=" * 60)
    
    from openrec.modeling.encoders.svtrv2_mamba import DASSMBlock
    
    B, C, H, W = 2, 128, 8, 32
    x = torch.randn(B, C, H, W).cuda()
    
    block = DASSMBlock(dim=C, head_dim=16).cuda()
    out = block(x)
    
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert x.shape == out.shape, "Shape mismatch!"
    print("  ✅ DASSMBlock 테스트 통과")
    return True


def test_svtrv2_mamba():
    """전체 모델 테스트"""
    print("\n" + "=" * 60)
    print("4. SVTRv2Mamba 전체 모델 테스트")
    print("=" * 60)
    
    from openrec.modeling.encoders.svtrv2_mamba import SVTRv2Mamba
    
    model = SVTRv2Mamba(
        max_sz=[32, 128],
        in_channels=3,
        out_channels=256,
        dims=[128, 256, 384],
        depths=[6, 6, 6],
        num_heads=[4, 8, 12],
        head_dim=16,
        mlp_ratio=4,
        mixer=[
            ['Conv', 'Conv', 'Conv', 'Conv', 'Conv', 'Conv'],
            ['Conv', 'Conv', 'FMamba', 'Mamba', 'Mamba', 'Mamba'],
            ['Mamba', 'Mamba', 'Mamba', 'Mamba', 'Mamba', 'Mamba']
        ],
        num_convs=[[2, 2, 2, 2, 2, 2], [2, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        kernel_sizes=[[3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3]],
        sub_k=[[1, 1], [2, 1], [1, 1]],
        use_pos_embed=False,
        last_stage=False,
        feat2d=True,
    ).cuda()
    
    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Forward
    model.eval()
    x = torch.randn(2, 3, 32, 128).cuda()
    
    with torch.no_grad():
        out = model(x)
    
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print("  ✅ SVTRv2Mamba 테스트 통과")
    return True


def test_gradient():
    """Gradient 테스트"""
    print("\n" + "=" * 60)
    print("5. Gradient 테스트")
    print("=" * 60)
    
    from openrec.modeling.encoders.svtrv2_mamba import SVTRv2Mamba
    
    model = SVTRv2Mamba(
        dims=[64, 128, 192],
        depths=[2, 2, 2],
        num_heads=[2, 4, 6],
        head_dim=16,
        mixer=[
            ['Conv', 'Conv'],
            ['Conv', 'Mamba'],
            ['Mamba', 'Mamba']
        ],
        sub_k=[[1, 1], [2, 1], [1, 1]],
        feat2d=True,
    ).cuda()
    
    model.train()
    x = torch.randn(2, 3, 32, 64, requires_grad=True).cuda()
    
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    print(f"  Input gradient norm: {x.grad.norm().item():.6f}")
    print("  ✅ Gradient 테스트 통과")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SVTRv2Mamba 테스트 (Fallback 없음, CUDA 필수)")
    print("=" * 60 + "\n")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        sys.exit(1)
    
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print()
    
    try:
        test_import()
        test_dassm()
        test_dassm_block()
        test_svtrv2_mamba()
        test_gradient()
        
        print("\n" + "=" * 60)
        print("✅ 모든 테스트 통과!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
