"""
Validation script for scientific presentations.
Checks slides against design principles and accessibility standards.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
from pptx import Presentation
from pptx.util import Pt


class PresentationValidator:
    """Validates scientific presentations against design standards."""
    
    # Design standards
    MIN_FONT_SIZE = 16  # Points
    MIN_TITLE_SIZE = 32  # Points
    PREFERRED_BODY_SIZE = 18  # Points
    PREFERRED_TITLE_SIZE = 36  # Points
    MIN_LINE_WIDTH = 1  # Points
    MIN_CONTRAST_RATIO = 4.5  # WCAG AA standard
    MAX_BULLETS_PER_SLIDE = 7
    PREFERRED_ASPECT_RATIO = (16, 9)
    
    # Colorblind-safe palettes
    WONG_PALETTE = [
        '#000000', '#E69F00', '#56B4E9', '#009E73',
        '#F0E442', '#0072B2', '#D55E00', '#CC79A7'
    ]
    
    def __init__(self, pptx_path: str):
        """
        Initialize validator with presentation path.
        
        Parameters
        ----------
        pptx_path : str
            Path to PowerPoint presentation file
        """
        self.path = Path(pptx_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Presentation not found: {pptx_path}")
        
        self.prs = Presentation(str(self.path))
        self.issues = []
        self.warnings = []
        self.successes = []
    
    def validate_all(self) -> Dict[str, List[str]]:
        """
        Run all validation checks.
        
        Returns
        -------
        dict
            Dictionary with 'issues', 'warnings', and 'successes' lists
        """
        self.validate_aspect_ratio()
        self.validate_slide_count()
        
        for slide_idx, slide in enumerate(self.prs.slides, 1):
            self.validate_slide(slide, slide_idx)
        
        return {
            'issues': self.issues,
            'warnings': self.warnings,
            'successes': self.successes
        }
    
    def validate_aspect_ratio(self):
        """Check if presentation uses 16:9 aspect ratio."""
        width = self.prs.slide_width
        height = self.prs.slide_height
        ratio = width / height
        
        expected_ratio = self.PREFERRED_ASPECT_RATIO[0] / self.PREFERRED_ASPECT_RATIO[1]
        
        if abs(ratio - expected_ratio) > 0.01:
            self.issues.append(
                f"❌ Aspect ratio is {ratio:.2f}:1, should be 16:9 (1.78:1)"
            )
        else:
            self.successes.append("✓ Correct aspect ratio (16:9)")
    
    def validate_slide_count(self):
        """Check for reasonable slide count."""
        slide_count = len(self.prs.slides)
        
        if slide_count == 0:
            self.issues.append("❌ Presentation has no slides")
        elif slide_count > 100:
            self.warnings.append(
                f"⚠️  Very long presentation ({slide_count} slides). "
                "Consider splitting or condensing."
            )
        else:
            self.successes.append(f"✓ Reasonable slide count ({slide_count} slides)")
    
    def validate_slide(self, slide, slide_idx: int):
        """
        Validate a single slide.
        
        Parameters
        ----------
        slide : pptx.slide.Slide
            Slide object to validate
        slide_idx : int
            Slide number (1-indexed)
        """
        prefix = f"Slide {slide_idx}"
        
        # Check for title
        has_title = False
        title_font_size = None
        
        for shape in slide.shapes:
            if shape.has_text_frame:
                # Check if this is a title (usually first text box, larger font)
                if shape.text_frame.text.strip():
                    for paragraph in shape.text_frame.paragraphs:
                        for run in paragraph.runs:
                            if run.font.size:
                                font_size_pt = run.font.size.pt
                                
                                # Likely a title if ≥28pt and near top
                                if font_size_pt >= 28 and shape.top < self.prs.slide_height / 4:
                                    has_title = True
                                    title_font_size = font_size_pt
                                    
                                    # Check title size
                                    if font_size_pt < self.MIN_TITLE_SIZE:
                                        self.issues.append(
                                            f"❌ {prefix}: Title font size {font_size_pt:.0f}pt "
                                            f"is below minimum {self.MIN_TITLE_SIZE}pt"
                                        )
                                    elif font_size_pt < self.PREFERRED_TITLE_SIZE:
                                        self.warnings.append(
                                            f"⚠️  {prefix}: Title font size {font_size_pt:.0f}pt "
                                            f"is below preferred {self.PREFERRED_TITLE_SIZE}pt"
                                        )
                                
                                # Check body text
                                if font_size_pt < self.MIN_FONT_SIZE:
                                    self.issues.append(
                                        f"❌ {prefix}: Font size {font_size_pt:.0f}pt "
                                        f"is below minimum {self.MIN_FONT_SIZE}pt"
                                    )
                                elif font_size_pt < self.PREFERRED_BODY_SIZE:
                                    self.warnings.append(
                                        f"⚠️  {prefix}: Font size {font_size_pt:.0f}pt "
                                        f"is below preferred {self.PREFERRED_BODY_SIZE}pt"
                                    )
        
        if not has_title:
            self.warnings.append(
                f"⚠️  {prefix}: No clear title found (or title font <28pt)"
            )
        
        # Check bullet count
        bullet_count = self.count_bullets(slide)
        if bullet_count > self.MAX_BULLETS_PER_SLIDE:
            self.issues.append(
                f"❌ {prefix}: {bullet_count} bullets exceeds maximum "
                f"{self.MAX_BULLETS_PER_SLIDE} (cognitive overload)"
            )
        
        # Check for text density
        text_length = sum(len(shape.text) for shape in slide.shapes if shape.has_text_frame)
        if text_length > 500:
            self.warnings.append(
                f"⚠️  {prefix}: High text density ({text_length} characters). "
                "Consider using visuals or splitting slide."
            )
    
    def count_bullets(self, slide) -> int:
        """
        Count bullet points on a slide.
        
        Parameters
        ----------
        slide : pptx.slide.Slide
            Slide to analyze
        
        Returns
        -------
        int
            Number of bullet points
        """
        bullet_count = 0
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    # Check if paragraph has bullet formatting
                    if paragraph.level is not None or paragraph.text.strip().startswith(('•', '-', '*')):
                        if paragraph.text.strip():
                            bullet_count += 1
        return bullet_count
    
    def print_report(self):
        """Print validation report to console."""
        print("\n" + "="*70)
        print(f"PRESENTATION VALIDATION REPORT: {self.path.name}")
        print("="*70 + "\n")
        
        results = self.validate_all()
        
        if results['issues']:
            print("🚨 CRITICAL ISSUES (Must Fix):")
            print("-" * 70)
            for issue in results['issues']:
                print(f"  {issue}")
            print()
        
        if results['warnings']:
            print("⚠️  WARNINGS (Should Fix):")
            print("-" * 70)
            for warning in results['warnings']:
                print(f"  {warning}")
            print()
        
        if results['successes']:
            print("✓ PASSING CHECKS:")
            print("-" * 70)
            for success in results['successes']:
                print(f"  {success}")
            print()
        
        # Summary
        print("="*70)
        total_checks = len(results['issues']) + len(results['warnings']) + len(results['successes'])
        print(f"SUMMARY: {len(results['successes'])}/{total_checks} checks passed")
        
        if not results['issues']:
            print("✓ No critical issues found!")
        else:
            print(f"⚠️  {len(results['issues'])} critical issue(s) to fix")
        
        if results['warnings']:
            print(f"⚠️  {len(results['warnings'])} warning(s) to address")
        
        print("="*70 + "\n")
        
        return len(results['issues']) == 0


def main():
    """Command-line interface for validation."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_presentation.py <presentation.pptx>")
        sys.exit(1)
    
    pptx_path = sys.argv[1]
    
    try:
        validator = PresentationValidator(pptx_path)
        passed = validator.print_report()
        
        # Exit with appropriate code
        sys.exit(0 if passed else 1)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
