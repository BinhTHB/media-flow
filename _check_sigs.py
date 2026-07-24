import json

# Read existing to understand structure
gt = json.loads(open('artifacts/round26/gt_v2_2.json', encoding='utf-8').read())
pos = [e for e in gt['entries'] if e['is_subtitle']]
neg = [e for e in gt['entries'] if not e['is_subtitle']]
print('GT V2.2: %d positive, %d negative' % (len(pos), len(neg)))

# Check R1MotionCoupling constructor signature and evaluate method
import inspect
from src.infrastructure.benchmark.hybrid_subtitle.r1_motion import R1MotionCoupling, R1Result
sig = inspect.signature(R1MotionCoupling.__init__)
print('R1MotionCoupling.__init__ signature: %s' % sig)
ev_sig = inspect.signature(R1MotionCoupling.evaluate)
print('R1MotionCoupling.evaluate signature: %s' % ev_sig)
print('R1Result fields: rejected=%s, neutral=%s' % (hasattr(R1Result, 'rejected'), hasattr(R1Result, 'neutral')))

# Check R2
from src.infrastructure.benchmark.hybrid_subtitle.r2_language import R2LanguageDetector
r2_sig = inspect.signature(R2LanguageDetector.__init__)
print('R2LanguageDetector.__init__ signature: %s' % r2_sig)
det_sig = inspect.signature(R2LanguageDetector.detect)
print('R2LanguageDetector.detect signature: %s' % det_sig)

# Check R3
from src.infrastructure.benchmark.hybrid_subtitle.behavior_discriminator import R3StaticDiscriminator
r3_sig = inspect.signature(R3StaticDiscriminator.__init__)
print('R3StaticDiscriminator.__init__ signature: %s' % r3_sig)

# Check available attributes on R1Result
r = R1Result.__dataclass_fields__
print('R1Result fields: %s' % list(r.keys()))
